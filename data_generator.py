import numpy as np
import os
import re
from PIL import Image, ImageEnhance
import copy
import openpyxl
import zipfile
from global_var import myModelConfig


class pasi_data():

    def __init__(self):

        self.patient_dict = self.build_patient_dict()

    def cal_distance(self, x1, x2, y1, y2):
        # 计算在input image上两点的距离

        distance = np.sqrt(np.sum(np.square(np.array([x1, y1]) - np.array([x2, y2]))))

        return distance


    def gaussian_kernel(self, x, y, sigma):
        val = np.exp(- (x * x + y * y) / (2 * sigma * sigma))
        return val


    def single_disease_map_label(self, disease_map_size, RF_size, stride, gt_box, thr):
        '''

        :param disease_map_size:  单个disease map的size，[height, width]
        :param gt_box: gt_box, [[x1,y1,x2,y2],[x1,y1,x2,y2],.......],shape=[k,4]
        :return: single_disease_map_label

        mapped_x, mapped_t  = (⌊s⌋+xs,􏰁s􏰂+ys)
        '''

        label_map = np.zeros(shape=disease_map_size)

        row = len(label_map)
        col = len(label_map[0])

        if len(gt_box) == 0:
            return label_map

        # 先将box转为中心点，再转换为feature map上的对应点, 并记录该中心对应的gaussian kernel的radius，取其1/3作为sigma。gaussian kernel = exp(- (x**2 + y**2)/(2*  sigma**2))
        # 且只对该box的radius范围内的点计算该值
        center_list = []

        for box in gt_box:
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / (2 * stride))
            center_y = int((y1 + y2) / (2 * stride))

            # 求得该feature map上两点的距离
            radius = self.cal_distance(x1, x2, y1, y2) / (2 * stride)

            center_list.append([center_x, center_y, radius])

        for y in range(row):
            for x in range(col):
                for center in center_list:

                    center_x, center_y, radius = center
                    if self.cal_distance(x, center_x, y, center_y) <= radius:
                        label_map[y][x] += self.gaussian_kernel(abs(x - center_x), abs(y - center_y), (1 / 3) * radius)

        return label_map


    def get_disease_map_label(self, gt_box):
        # disease_map_size, RF_size, stride, gt_box, thr
        disease_map_size = [[100, 128], [50, 64], [25, 32], [13, 16], [7, 8]]
        RF_list = [117, 325, 549, 741, 1125]
        stride_list = [8, 16, 32, 64, 128]

        map_label_list = []
        for index in range(5):
            map_label = self.single_disease_map_label(disease_map_size[index], RF_list[index],
                                                 stride_list[index], gt_box, 0.3)

            map_label = np.reshape(map_label, [disease_map_size[index][0], disease_map_size[index][1], 1])
            map_label_list.append(map_label)

        return map_label_list


    def valid_data_preprocessing(self, origin_img, gt_box):
        if len(gt_box) == 0:
            # 如果未检测到gt_box, 图像不变

            img = origin_img


        else:
            # 如果检测到box，需要将box变形
            img = origin_img
            w = img.size[0]
            h = img.size[1]

            for i in range(len(gt_box)):
                # 将左上右下坐标转化为中心点+wh坐标, 注意这里的坐标点都是未归一化的
                center_x = (gt_box[i][0] + gt_box[i][2]) / 2 - 1
                center_y = (gt_box[i][1] + gt_box[i][3]) / 2 - 1

                box_w = gt_box[i][2] - gt_box[i][0]
                box_h = gt_box[i][3] - gt_box[i][1]

                gt_box[i] = [center_x, center_y, box_w, box_h]

                gt_box[i][0] /= w
                gt_box[i][1] /= h
                gt_box[i][2] /= w
                gt_box[i][3] /= h

        img = img.resize((1024, 800))
        img = np.array(img)
        img = img / 255.0

        # 将坐标转化为resize过后的左上右下坐标
        for i in range(len(gt_box)):
            gt_box[i][0] *= 1024
            gt_box[i][1] *= 800
            gt_box[i][2] *= 1024
            gt_box[i][3] *= 800

            left_top_x = max(int(gt_box[i][0] - gt_box[i][2] / 2), 0)
            left_top_y = max(int(gt_box[i][1] - gt_box[i][3] / 2), 0)

            right_down_x = min(int(gt_box[i][0] + gt_box[i][2] / 2), 1024)
            right_down_y = min(int(gt_box[i][1] + gt_box[i][3] / 2), 800)

            gt_box[i] = [left_top_x, left_top_y, right_down_x, right_down_y]
            assert right_down_x > left_top_x and right_down_y > left_top_y, gt_box[i]

        return img, gt_box


    def train_data_preprocessing(self, origin_img, gt_box):
        if len(gt_box) == 0:
            # 如果未检测到gt_box, 按70%-100%的范围随机crop图像, 当应用了gt-box的时候该步骤取消
            ratio1 = np.random.uniform(0, 0.15)
            shape = (origin_img.size)
            w = shape[0]
            h = shape[1]

            x1 = int(ratio1 * w)
            y1 = int(ratio1 * h)

            ratio2 = np.random.uniform(0, 0.15)
            x2 = int(w - ratio2 * w)
            y2 = int(h - ratio2 * h)

            box = (x1, y1, x2, y2)
            img = origin_img.crop(box)


        else:
            # 按gt-box的边界随机crop图像，即每次crop不会crop到gt-box
            shape = (origin_img.size)
            w = shape[0]
            h = shape[1]

            x1_list = [elem[0] for elem in gt_box]
            y1_list = [elem[1] for elem in gt_box]
            x2_list = [elem[2] for elem in gt_box]
            y2_list = [elem[3] for elem in gt_box]

            min_x = min(min(x1_list), int(w * 0.15))
            max_x = max(max(x2_list), int(w * 0.85))

            min_y = min(min(y1_list), int(h * 0.15))
            max_y = max(max(y2_list), int(h * 0.85))

            x1 = np.random.uniform(0, min_x)
            x2 = np.random.uniform(max_x, w)

            y1 = np.random.uniform(0, min_y)
            y2 = np.random.uniform(max_y, h)

            box = (x1, y1, x2, y2)

            img = origin_img.crop(box)

            crop_w = img.size[0]
            crop_h = img.size[1]

            for i in range(len(gt_box)):
                gt_box[i][0] -= (x1 - 1)
                gt_box[i][1] -= (y1 - 1)
                gt_box[i][2] -= (x1 - 1)
                gt_box[i][3] -= (y1 - 1)

                # 将左上右下坐标转化为中心点+wh坐标, 注意这里的坐标点都是未归一化的
                center_x = (gt_box[i][0] + gt_box[i][2]) / 2 - 1
                center_y = (gt_box[i][1] + gt_box[i][3]) / 2 - 1

                box_w = gt_box[i][2] - gt_box[i][0]
                box_h = gt_box[i][3] - gt_box[i][1]

                gt_box[i] = [center_x, center_y, box_w, box_h]

                gt_box[i][0] /= crop_w
                gt_box[i][1] /= crop_h
                gt_box[i][2] /= crop_w
                gt_box[i][3] /= crop_h

        img = img.resize((1024, 800))

        if np.random.uniform(0, 1) > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for i in range(len(gt_box)):
                gt_box[i][0] = 1 - gt_box[i][0]

        if np.random.uniform(0, 1) > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            for i in range(len(gt_box)):
                gt_box[i][1] = 1 - gt_box[i][1]

        if np.random.uniform(0, 1) > 0.5:
            img = ImageEnhance.Brightness(img)
            brightness = np.random.uniform(0.8, 1.2)
            img = img.enhance(brightness)

        if np.random.uniform(0, 1) > 0.5:
            img = ImageEnhance.Color(img)
            brightness = np.random.uniform(0.8, 1.2)
            img = img.enhance(brightness)

        img = np.array(img)
        img = img / 255.0

        # 将坐标转化为resize过后的左上右下坐标
        for i in range(len(gt_box)):
            gt_box[i][0] *= 1024
            gt_box[i][1] *= 800
            gt_box[i][2] *= 1024
            gt_box[i][3] *= 800

            left_top_x = max(int(gt_box[i][0] - gt_box[i][2] / 2), 0)
            left_top_y = max(int(gt_box[i][1] - gt_box[i][3] / 2), 0)

            right_down_x = min(int(gt_box[i][0] + gt_box[i][2] / 2), 1024)
            right_down_y = min(int(gt_box[i][1] + gt_box[i][3] / 2), 800)

            gt_box[i] = [left_top_x, left_top_y, right_down_x, right_down_y]
            assert right_down_x > left_top_x and right_down_y > left_top_y, gt_box[i]

        return img, gt_box


    def build_patient_dict(self):

        excel_dir = myModelConfig.excel_path

        if not excel_dir:
            print("[pasi_data] excel_path is empty in config; proceeding without patient metadata.")
            return {}

        candidate_paths = [excel_dir]
        config_dir = getattr(myModelConfig, "config_dir", None)
        if config_dir and not os.path.isabs(excel_dir):
            candidate_paths.append(os.path.join(config_dir, excel_dir))

        resolved_excel = None
        for path in candidate_paths:
            if os.path.isfile(path):
                resolved_excel = path
                break

        if not resolved_excel:
            raise FileNotFoundError(
                "Patient metadata Excel file not found. "
                f"Tried: {candidate_paths}. "
                "Update config.excel_path to point to a valid .xlsx file."
            )

        try:
            wb = openpyxl.load_workbook(resolved_excel)
        except (openpyxl.utils.exceptions.InvalidFileException, zipfile.BadZipFile):
            raise ValueError(
                f"Patient metadata file '{resolved_excel}' is not a valid .xlsx workbook. "
                "Ensure the file is saved in Excel format (ZIP-based .xlsx) and retry."
            )

        sheet_name = "Sheet1"
        if sheet_name not in wb.sheetnames:
            raise ValueError(
                f"Workbook '{excel_dir}' does not contain a sheet named '{sheet_name}'. "
                "Update the sheet name in data_generator.build_patient_dict if needed."
            )

        sheet = wb[sheet_name]
        patient_dict = dict()
        patient_list = sheet["A"][1:]

        for elem in patient_list:
            if isinstance(elem.value, str):
                patient_dict[elem.value] = dict()

        part_dict = {"D": "头颈部", "I": "躯干部", "N": "上肢", "S": "下肢", "X": "PASI"}

        # 构建patient_dict
        # 一个嵌套的dict，最外层key=patient name， value为不同part
        # 次内层不同part dict的key为part name或者pasi总分， value为4个分值dict或者pasi总分值
        # 最内层不同分值 dict的key为分值名， value为分值

        # 构建顺序为根据不同部位构建
        for part_index in part_dict.keys():

            # 判断当前处理的part不是pasi总分
            if part_index != "X":

                # 根据index取得该part的4个不同分值对应在excel里的位置
                part_name = part_dict[part_index]
                part_area_index = chr(ord(part_index) + 1)
                part_ery_index = chr(ord(part_index) + 2)
                part_sca_index = chr(ord(part_index) + 3)
                part_ind_index = chr(ord(part_index) + 4)

                # 取得该part的所有病人的4个不同分值
                part_area = sheet[part_area_index][1:]
                part_ery = sheet[part_ery_index][1:]
                part_sca = sheet[part_sca_index][1:]
                part_ind = sheet[part_ind_index][1:]

                # 将不同病人的不同分值加入到对应的part_area字典里
                for index, area in enumerate(part_area):

                    patient_name = patient_list[index].value
                    cur_area = part_area[index].value
                    cur_ery = part_ery[index].value
                    cur_sca = part_sca[index].value
                    cur_ind = part_ind[index].value

                    # 如果面积为0，那么其他几个分值一定为0，直接不处理
                    try:
                        if isinstance(cur_area, (int, float)):
                            cur_area = float(cur_area)
                        elif isinstance(cur_area, str):
                            cur_area = float(cur_area.strip())

                        if cur_area != 0:
                            patient_dict[patient_name][part_name] = dict()
                            patient_dict[patient_name][part_name]["area"] = float(cur_area) / 10.0
                            patient_dict[patient_name][part_name]["erythema"] = float(cur_ery)
                            patient_dict[patient_name][part_name]["scale"] = float(cur_sca)
                            patient_dict[patient_name][part_name]["induration"] = float(cur_ind)

                    except:
                        continue

            else:

                # 当前part如果是pasi总分，那就在该病人的dict中加入一个pasi dict
                part_pasi = sheet["X"][1:]

                for index, pasi in enumerate(part_pasi):
                    patient_name = patient_list[index].value
                    try:
                        patient_dict[patient_name]["pasi"] = float(part_pasi[index].value)
                    except:
                        continue

        return patient_dict


    def train_generator(self, batch_size=1):

        root_dir = myModelConfig.data_root
        train_file = myModelConfig.train_txt_file

        # 构建训练数据 list， 其中存放训练病人名
        train_patient_list = []
        # patient_list = os.listdir(root_dir)
        #
        with open(train_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                train_patient_list.append(line)
        train_patient_list = [elem for elem in train_patient_list if os.path.isdir(os.path.join(root_dir, elem))]

        img_list = []
        txt_list = []

        # 按patient_list中的顺序将病人图片名和box标签txt名分别存入list，注意顺序是对应的
        for elem in train_patient_list:
            patient_path = os.path.join(root_dir, elem)
            _cur_file_list = os.listdir(patient_path)
            _cur_img_list = [file for file in _cur_file_list if os.path.splitext(file)[-1].lower() == '.jpg']
            _cur_txt_list = [file[:-3] + 'txt' for file in _cur_img_list]

            cur_img_list = [os.path.join(patient_path, file) for file in _cur_img_list]
            cur_txt_list = [os.path.join(patient_path, file) for file in _cur_txt_list]

            img_list.extend(cur_img_list)
            txt_list.extend(cur_txt_list)

        # 将txt标签中的box解析为坐标值
        box_list = []
        for elem in txt_list:
            cur_box = []
            with open(elem, 'r') as f:
                for line in f.readlines():
                    box = line.strip().split(',')
                    box = [int(cord) for cord in box]

                    # 将顺序y1 y2 x1 x2 转化为x1 y1 x2 y2
                    # box[0], box[1], box[2], box[3] = box[2], box[0], box[3], box[1]
                    assert len(box) == 4, "failed at %s" % elem
                    assert box[2] > box[0] and box[3] > box[1], elem

                    # 将左上右下坐标转化为中心点+wh坐标, 注意这里的坐标点都是未归一化的
                    # center_x = int((box[0]+box[2])/2)
                    # center_y = int((box[1]+box[3])/2)
                    #
                    # w = box[2]-box[0]
                    # h = box[3]-box[1]
                    #
                    # center_box = [center_x, center_y, w, h]

                    cur_box.append(box)

            box_list.append(cur_box)

        # 得到最终的训练数据zip，之后每个训练sample就是zip中的一个item，分别为图片名和该图片的box
        zip_list = list(zip(img_list, box_list))

        while True:

            zip_copy = copy.deepcopy(zip_list)
            np.random.shuffle(zip_copy)

            img_list_epoch = [elem[0] for elem in zip_copy]
            box_list_epoch = [elem[1] for elem in zip_copy]

            batch_img_a = []
            batch_img_b = []

            batch_scoreA = []
            batch_scoreB = []
            batch_scoreSiam = []

            batch_locate_map_p3_a = []
            batch_locate_map_p4_a = []
            batch_locate_map_p5_a = []
            batch_locate_map_p6_a = []
            batch_locate_map_p7_a = []

            batch_locate_map_p3_b = []
            batch_locate_map_p4_b = []
            batch_locate_map_p5_b = []
            batch_locate_map_p6_b = []
            batch_locate_map_p7_b = []

            while len(img_list_epoch) != 0:

                if len(img_list_epoch) >= 2:

                    box1 = box_list_epoch.pop()
                    img1 = img_list_epoch.pop()
                    img1_basename = os.path.basename(img1)

                    name1 = img1_basename[:-4]

                    try:
                        patient_name1, part_name1 = tuple(re.split('-|-|/| |\n|\t', name1))
                    except:
                        print("parse error for patient name {}".format(name1))
                        continue

                    if patient_name1 not in self.patient_dict.keys():
                        print("patient {} not in patient dict".format(patient_name1))
                        continue
                    if part_name1 not in self.patient_dict[patient_name1].keys():
                        print("part {} not in patient {}'s dict".format(part_name1, patient_name1))
                        continue

                    try:
                        score1_area = self.patient_dict[patient_name1][part_name1]["area"]
                    except:
                        print("area not in patient {}'s {} dict".format(patient_name1, part_name1))
                        score1_area = 0.0

                    try:
                        score1_ery = self.patient_dict[patient_name1][part_name1]["erythema"]
                    except:
                        print("ery not in patient {}'s {} dict".format(patient_name1, part_name1))
                        score1_ery = 0.0

                    try:
                        score1_sca = self.patient_dict[patient_name1][part_name1]["scale"]
                    except:
                        print("sca not in patient {}'s {} dict".format(patient_name1, part_name1))
                        score1_sca = 0.0

                    try:
                        score1_ind = self.patient_dict[patient_name1][part_name1]["induration"]
                    except:
                        print("ind not in patient {}'s {} dict".format(patient_name1, part_name1))
                        score1_ind = 0.0

                    try:
                        score1_pasi = self.patient_dict[patient_name1]["pasi"]
                    except:
                        print("pasi not in patient {}'s dict".format(patient_name1))
                        continue

                    score1 = [score1_area, score1_ery, score1_sca, score1_ind, score1_pasi]
                    _img1 = Image.open(img1)

                    try:
                        img1, box1 = self.train_data_preprocessing(_img1, box1)
                    except:
                        raise

                    map_label1 = self.get_disease_map_label(box1)

                    box2 = box_list_epoch.pop()
                    img2 = img_list_epoch.pop()
                    img2_basename = os.path.basename(img2)

                    name2 = img2_basename[:-4]

                    try:
                        patient_name2, part_name2 = tuple(re.split('-|-|/| |\n|\t', name2))
                    except:
                        continue

                    if patient_name2 not in self.patient_dict.keys():
                        print("patient {} not in patient dict".format(patient_name2))
                        continue
                    if part_name2 not in self.patient_dict[patient_name2].keys():
                        print("part {} not in patient {}'s dict".format(part_name2, patient_name2))
                        continue

                    try:
                        score2_area = self.patient_dict[patient_name2][part_name2]["area"]
                    except:
                        print("area not in patient {}'s {} dict".format(patient_name2, part_name2))
                        score2_area = 0.0

                    try:
                        score2_ery = self.patient_dict[patient_name2][part_name2]["erythema"]
                    except:
                        print("ery not in patient {}'s {} dict".format(patient_name2, part_name2))
                        score2_ery = 0.0

                    try:
                        score2_sca = self.patient_dict[patient_name2][part_name2]["scale"]
                    except:
                        print("sca not in patient {}'s {} dict".format(patient_name2, part_name2))
                        score2_sca = 0.0

                    try:
                        score2_ind = self.patient_dict[patient_name2][part_name2]["induration"]
                    except:
                        print("ind not in patient {}'s {} dict".format(patient_name2, part_name2))
                        score2_ind = 0.0

                    try:
                        score2_pasi = self.patient_dict[patient_name2]["pasi"]
                    except:
                        print("pasi not in patient {}'s dict".format(patient_name2))
                        continue

                    score2 = [score2_area, score2_ery, score2_sca, score2_ind, score2_pasi]

                    _img2 = Image.open(img2)

                    try:
                        img2, box2 = self.train_data_preprocessing(_img2, box2)
                    except:
                        raise

                    map_label2 = self.get_disease_map_label(box2)

                    # yield_img = [np.array(img1), np.array(img2)]
                    # yield_label = [score1, score2, [abs(elem[0] - elem[1]) for elem in zip(score1, score2)]]

                    # [yield_label.append(elem) for elem in map_label1]
                    # [yield_label.append(elem) for elem in map_label2]

                    batch_img_a.append(np.array(img1))
                    batch_img_b.append(np.array(img2))

                    batch_scoreA.append(score1)
                    batch_scoreB.append(score2)
                    batch_scoreSiam.append([abs(elem[0] - elem[1]) for elem in zip(score1, score2)])

                    batch_locate_map_p3_a.append(map_label1[0])
                    batch_locate_map_p4_a.append(map_label1[1])
                    batch_locate_map_p5_a.append(map_label1[2])
                    batch_locate_map_p6_a.append(map_label1[3])
                    batch_locate_map_p7_a.append(map_label1[4])

                    batch_locate_map_p3_b.append(map_label2[0])
                    batch_locate_map_p4_b.append(map_label2[1])
                    batch_locate_map_p5_b.append(map_label2[2])
                    batch_locate_map_p6_b.append(map_label2[3])
                    batch_locate_map_p7_b.append(map_label2[4])

                    if len(batch_img_a) == batch_size:
                        # yield ({"input_a": img1, "input_b": img2},{"scoreA": score1, "scoreB": score2, "scoreSiam": abs(score1 - score2)})
                        yield [np.array(batch_img_a), np.array(batch_img_b)], [np.array(batch_scoreA),
                                                                               np.array(batch_scoreB),
                                                                               np.array(batch_scoreSiam),

                                                                               np.array(batch_locate_map_p3_a),
                                                                               np.array(batch_locate_map_p4_a),
                                                                               np.array(batch_locate_map_p5_a),
                                                                               np.array(batch_locate_map_p6_a),
                                                                               np.array(batch_locate_map_p7_a),

                                                                               np.array(batch_locate_map_p3_b),
                                                                               np.array(batch_locate_map_p4_b),
                                                                               np.array(batch_locate_map_p5_b),
                                                                               np.array(batch_locate_map_p6_b),
                                                                               np.array(batch_locate_map_p7_b)]

                        batch_img_a.clear()
                        batch_img_b.clear()

                        batch_scoreA.clear()
                        batch_scoreB.clear()
                        batch_scoreSiam.clear()
                        batch_locate_map_p3_a.clear()
                        batch_locate_map_p4_a.clear()
                        batch_locate_map_p5_a.clear()
                        batch_locate_map_p6_a.clear()
                        batch_locate_map_p7_a.clear()
                        batch_locate_map_p3_b.clear()
                        batch_locate_map_p4_b.clear()
                        batch_locate_map_p5_b.clear()
                        batch_locate_map_p6_b.clear()
                        batch_locate_map_p7_b.clear()
                        # yield img1, img2, score1, score2, abs(score1-score2), map_label1[:], map_label2[:]
                    else:
                        continue
                else:
                    break

            if len(batch_img_a) + len(batch_img_b) > 0:
                yield [np.array(batch_img_a), np.array(batch_img_b)], [np.array(batch_scoreA),
                                                                       np.array(batch_scoreB),
                                                                       np.array(batch_scoreSiam),

                                                                       np.array(batch_locate_map_p3_a),
                                                                       np.array(batch_locate_map_p4_a),
                                                                       np.array(batch_locate_map_p5_a),
                                                                       np.array(batch_locate_map_p6_a),
                                                                       np.array(batch_locate_map_p7_a),

                                                                       np.array(batch_locate_map_p3_b),
                                                                       np.array(batch_locate_map_p4_b),
                                                                       np.array(batch_locate_map_p5_b),
                                                                       np.array(batch_locate_map_p6_b),
                                                                       np.array(batch_locate_map_p7_b)]


    def valid_generator(self, batch_size=1):

        root_dir = myModelConfig.data_root
        val_file = myModelConfig.val_txt_file

        # 构建训练数据 list， 其中存放训练病人名
        val_patient_list = []
        # patient_list = os.listdir(root_dir)
        #
        with open(val_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                val_patient_list.append(line)
        val_patient_list = [elem for elem in val_patient_list if os.path.isdir(os.path.join(root_dir, elem))]

        img_list = []
        txt_list = []

        # 按patient_list中的顺序将病人图片名和box标签txt名分别存入list，注意顺序是对应的
        for elem in val_patient_list:
            patient_path = os.path.join(root_dir, elem)
            _cur_file_list = os.listdir(patient_path)
            _cur_img_list = [file for file in _cur_file_list if os.path.splitext(file)[-1].lower() == '.jpg']
            _cur_txt_list = [file[:-3] + 'txt' for file in _cur_img_list]

            cur_img_list = [os.path.join(patient_path, file) for file in _cur_img_list]
            cur_txt_list = [os.path.join(patient_path, file) for file in _cur_txt_list]

            img_list.extend(cur_img_list)
            txt_list.extend(cur_txt_list)

        # 将txt标签中的box解析为坐标值
        box_list = []
        for elem in txt_list:
            cur_box = []
            with open(elem, 'r') as f:
                for line in f.readlines():
                    box = line.strip().split(',')
                    box = [int(cord) for cord in box]

                    # 将顺序y1 y2 x1 x2 转化为x1 y1 x2 y2
                    # box[0], box[1], box[2], box[3] = box[2], box[0], box[3], box[1]
                    assert len(box) == 4, "failed at %s" % elem
                    assert box[2] > box[0] and box[3] > box[1], elem

                    # 将左上右下坐标转化为中心点+wh坐标, 注意这里的坐标点都是未归一化的
                    # center_x = int((box[0]+box[2])/2)
                    # center_y = int((box[1]+box[3])/2)
                    #
                    # w = box[2]-box[0]
                    # h = box[3]-box[1]
                    #
                    # center_box = [center_x, center_y, w, h]

                    cur_box.append(box)

            box_list.append(cur_box)

        # 得到最终的训练数据zip，之后每个训练sample就是zip中的一个item，分别为图片名和该图片的box
        zip_list = list(zip(img_list, box_list))

        while True:

            zip_copy = copy.deepcopy(zip_list)
            np.random.shuffle(zip_copy)

            img_list_epoch = [elem[0] for elem in zip_copy]
            box_list_epoch = [elem[1] for elem in zip_copy]

            batch_img_a = []
            batch_img_b = []

            batch_scoreA = []
            batch_scoreB = []
            batch_scoreSiam = []

            batch_locate_map_p3_a = []
            batch_locate_map_p4_a = []
            batch_locate_map_p5_a = []
            batch_locate_map_p6_a = []
            batch_locate_map_p7_a = []

            batch_locate_map_p3_b = []
            batch_locate_map_p4_b = []
            batch_locate_map_p5_b = []
            batch_locate_map_p6_b = []
            batch_locate_map_p7_b = []

            while len(img_list_epoch) != 0:

                if len(img_list_epoch) >= 2:

                    box1 = box_list_epoch.pop()
                    img1 = img_list_epoch.pop()
                    img1_basename = os.path.basename(img1)

                    name1 = img1_basename[:-4]

                    try:
                        patient_name1, part_name1 = tuple(re.split('-|-|/| |\n|\t', name1))
                    except:
                        print("parse error for patient name {}".format(name1))
                        continue

                    if patient_name1 not in self.patient_dict.keys():
                        print("patient {} not in patient dict".format(patient_name1))
                        continue
                    if part_name1 not in self.patient_dict[patient_name1].keys():
                        print("part {} not in patient {}'s dict".format(part_name1, patient_name1))
                        continue

                    try:
                        score1_area = self.patient_dict[patient_name1][part_name1]["area"]
                    except:
                        print("area not in patient {}'s {} dict".format(patient_name1, part_name1))
                        score1_area = 0.0

                    try:
                        score1_ery = self.patient_dict[patient_name1][part_name1]["erythema"]
                    except:
                        print("ery not in patient {}'s {} dict".format(patient_name1, part_name1))
                        score1_ery = 0.0

                    try:
                        score1_sca = self.patient_dict[patient_name1][part_name1]["scale"]
                    except:
                        print("sca not in patient {}'s {} dict".format(patient_name1, part_name1))
                        score1_sca = 0.0

                    try:
                        score1_ind = self.patient_dict[patient_name1][part_name1]["induration"]
                    except:
                        print("ind not in patient {}'s {} dict".format(patient_name1, part_name1))
                        score1_ind = 0.0

                    try:
                        score1_pasi = self.patient_dict[patient_name1]["pasi"]
                    except:
                        print("pasi not in patient {}'s dict".format(patient_name1))
                        continue

                    score1 = [score1_area, score1_ery, score1_sca, score1_ind, score1_pasi]
                    _img1 = Image.open(img1)

                    try:
                        img1, box1 = self.valid_data_preprocessing(_img1, box1)
                    except:
                        raise

                    map_label1 = self.get_disease_map_label(box1)

                    box2 = box_list_epoch.pop()
                    img2 = img_list_epoch.pop()
                    img2_basename = os.path.basename(img2)

                    name2 = img2_basename[:-4]

                    try:
                        patient_name2, part_name2 = tuple(re.split('-|-|/| |\n|\t', name2))
                    except:
                        continue

                    if patient_name2 not in self.patient_dict.keys():
                        print("patient {} not in patient dict".format(patient_name2))
                        continue
                    if part_name2 not in self.patient_dict[patient_name2].keys():
                        print("part {} not in patient {}'s dict".format(part_name2, patient_name2))
                        continue

                    try:
                        score2_area = self.patient_dict[patient_name2][part_name2]["area"]
                    except:
                        print("area not in patient {}'s {} dict".format(patient_name2, part_name2))
                        score2_area = 0.0

                    try:
                        score2_ery = self.patient_dict[patient_name2][part_name2]["erythema"]
                    except:
                        print("ery not in patient {}'s {} dict".format(patient_name2, part_name2))
                        score2_ery = 0.0

                    try:
                        score2_sca = self.patient_dict[patient_name2][part_name2]["scale"]
                    except:
                        print("sca not in patient {}'s {} dict".format(patient_name2, part_name2))
                        score2_sca = 0.0

                    try:
                        score2_ind = self.patient_dict[patient_name2][part_name2]["induration"]
                    except:
                        print("ind not in patient {}'s {} dict".format(patient_name2, part_name2))
                        score2_ind = 0.0

                    try:
                        score2_pasi = self.patient_dict[patient_name2]["pasi"]
                    except:
                        print("pasi not in patient {}'s dict".format(patient_name2))
                        continue

                    score2 = [score2_area, score2_ery, score2_sca, score2_ind, score2_pasi]

                    _img2 = Image.open(img2)

                    try:
                        img2, box2 = self.valid_data_preprocessing(_img2, box2)
                    except:
                        raise

                    map_label2 = self.get_disease_map_label(box2)

                    # yield_img = [np.array(img1), np.array(img2)]
                    # yield_label = [score1, score2, [abs(elem[0] - elem[1]) for elem in zip(score1, score2)]]

                    # [yield_label.append(elem) for elem in map_label1]
                    # [yield_label.append(elem) for elem in map_label2]

                    batch_img_a.append(np.array(img1))
                    batch_img_b.append(np.array(img2))

                    batch_scoreA.append(score1)
                    batch_scoreB.append(score2)
                    batch_scoreSiam.append([abs(elem[0] - elem[1]) for elem in zip(score1, score2)])

                    batch_locate_map_p3_a.append(map_label1[0])
                    batch_locate_map_p4_a.append(map_label1[1])
                    batch_locate_map_p5_a.append(map_label1[2])
                    batch_locate_map_p6_a.append(map_label1[3])
                    batch_locate_map_p7_a.append(map_label1[4])

                    batch_locate_map_p3_b.append(map_label2[0])
                    batch_locate_map_p4_b.append(map_label2[1])
                    batch_locate_map_p5_b.append(map_label2[2])
                    batch_locate_map_p6_b.append(map_label2[3])
                    batch_locate_map_p7_b.append(map_label2[4])

                    if len(batch_img_a) == batch_size:
                        # yield ({"input_a": img1, "input_b": img2},{"scoreA": score1, "scoreB": score2, "scoreSiam": abs(score1 - score2)})
                        yield [np.array(batch_img_a), np.array(batch_img_b)], [np.array(batch_scoreA),
                                                                               np.array(batch_scoreB),
                                                                               np.array(batch_scoreSiam),

                                                                               np.array(batch_locate_map_p3_a),
                                                                               np.array(batch_locate_map_p4_a),
                                                                               np.array(batch_locate_map_p5_a),
                                                                               np.array(batch_locate_map_p6_a),
                                                                               np.array(batch_locate_map_p7_a),

                                                                               np.array(batch_locate_map_p3_b),
                                                                               np.array(batch_locate_map_p4_b),
                                                                               np.array(batch_locate_map_p5_b),
                                                                               np.array(batch_locate_map_p6_b),
                                                                               np.array(batch_locate_map_p7_b)]

                        batch_img_a.clear()
                        batch_img_b.clear()

                        batch_scoreA.clear()
                        batch_scoreB.clear()
                        batch_scoreSiam.clear()
                        batch_locate_map_p3_a.clear()
                        batch_locate_map_p4_a.clear()
                        batch_locate_map_p5_a.clear()
                        batch_locate_map_p6_a.clear()
                        batch_locate_map_p7_a.clear()
                        batch_locate_map_p3_b.clear()
                        batch_locate_map_p4_b.clear()
                        batch_locate_map_p5_b.clear()
                        batch_locate_map_p6_b.clear()
                        batch_locate_map_p7_b.clear()
                        # yield img1, img2, score1, score2, abs(score1-score2), map_label1[:], map_label2[:]
                    else:
                        continue
                else:
                    break

            if len(batch_img_a) + len(batch_img_b) > 0:
                yield [np.array(batch_img_a), np.array(batch_img_b)], [np.array(batch_scoreA),
                                                                       np.array(batch_scoreB),
                                                                       np.array(batch_scoreSiam),

                                                                       np.array(batch_locate_map_p3_a),
                                                                       np.array(batch_locate_map_p4_a),
                                                                       np.array(batch_locate_map_p5_a),
                                                                       np.array(batch_locate_map_p6_a),
                                                                       np.array(batch_locate_map_p7_a),

                                                                       np.array(batch_locate_map_p3_b),
                                                                       np.array(batch_locate_map_p4_b),
                                                                       np.array(batch_locate_map_p5_b),
                                                                       np.array(batch_locate_map_p6_b),
                                                                       np.array(batch_locate_map_p7_b)]


    def test_generator(self, batch_size=1):

        root_dir = myModelConfig.data_root
        val_file = myModelConfig.val_txt_file

        # root_dir = "G:\\pasi\\pasi_detection"
        # val_file = "G:\\pasi\\val.txt"

        patient_list = []
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                patient_list.append(line)

        patient_list = [elem for elem in patient_list if os.path.isdir(os.path.join(root_dir, elem))]

        img_list = []
        txt_list = []

        for elem in patient_list:
            patient_path = os.path.join(root_dir, elem)
            _cur_file_list = os.listdir(patient_path)
            _cur_img_list = [file for file in _cur_file_list if os.path.splitext(file)[-1].lower() == '.jpg']
            _cur_txt_list = [file[:-3] + 'txt' for file in _cur_img_list]

            cur_img_list = [os.path.join(patient_path, file) for file in _cur_img_list]
            cur_txt_list = [os.path.join(patient_path, file) for file in _cur_txt_list]

            img_list.extend(cur_img_list)
            txt_list.extend(cur_txt_list)

        box_list = []
        for elem in txt_list:
            cur_box = []
            with open(elem, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    box = line.strip().split(',')
                    box = [int(cord) for cord in box]

                    # 将顺序y1 y2 x1 x2 转化为x1 y1 x2 y2
                    # box[0], box[1], box[2], box[3] = box[2], box[0], box[3], box[1]
                    assert len(box) == 4, "failed at %s" % elem
                    assert box[2] > box[0] and box[3] > box[1], elem

                    # 将左上右下坐标转化为中心点+wh坐标, 注意这里的坐标点都是未归一化的
                    # center_x = int((box[0]+box[2])/2)
                    # center_y = int((box[1]+box[3])/2)
                    #
                    # w = box[2]-box[0]
                    # h = box[3]-box[1]
                    #
                    # center_box = [center_x, center_y, w, h]

                    cur_box.append(box)

            box_list.append(cur_box)

        zip_list = list(zip(img_list, box_list))

        img_list_epoch = [elem[0] for elem in zip_list]
        box_list_epoch = [elem[1] for elem in zip_list]

        while len(img_list_epoch) != 0:

            if len(img_list_epoch) >= 2:
                box1 = box_list_epoch.pop()
                img1 = img_list_epoch.pop()
                img1_basename = os.path.basename(img1)

                name1 = img1_basename[:-4]

                try:
                    patient_name1, part_name1 = tuple(re.split('-|-|/| |\n|\t', name1))

                except:
                    continue

                if patient_name1 not in self.patient_dict.keys():
                    continue
                if part_name1 not in self.patient_dict[patient_name1].keys():
                    continue

                try:
                    score1_area = self.patient_dict[patient_name1][part_name1]["area"]
                except:
                    score1_area = 0.0

                try:
                    score1_ery = self.patient_dict[patient_name1][part_name1]["erythema"]
                except:
                    score1_ery = 0.0

                try:
                    score1_sca = self.patient_dict[patient_name1][part_name1]["scale"]
                except:
                    score1_sca = 0.0

                try:
                    score1_ind = self.patient_dict[patient_name1][part_name1]["induration"]
                except:
                    score1_ind = 0.0

                try:
                    score1_pasi = self.patient_dict[patient_name1]["pasi"]
                except:
                    continue

                score1 = [score1_area, score1_ery, score1_sca, score1_ind, score1_pasi]
                _img1 = Image.open(img1)

                try:
                    img1, box1 = self.valid_data_preprocessing(_img1, box1)
                except:
                    raise

                map_label1 = self.get_disease_map_label(box1)

                box2 = box_list_epoch.pop()
                img2 = img_list_epoch.pop()
                img2_basename = os.path.basename(img2)

                name2 = img2_basename[:-4]

                try:
                    patient_name2, part_name2 = tuple(re.split('-|-|/| |\n|\t', name2))
                except:
                    continue

                if patient_name2 not in self.patient_dict.keys():
                    continue
                if part_name2 not in self.patient_dict[patient_name2].keys():
                    continue

                try:
                    score2_area = self.patient_dict[patient_name2][part_name2]["area"] / 10.0
                except:
                    score2_area = 0.0

                try:
                    score2_ery = self.patient_dict[patient_name2][part_name2]["erythema"]
                except:
                    score2_ery = 0.0

                try:
                    score2_sca = self.patient_dict[patient_name2][part_name2]["scale"]
                except:
                    score2_sca = 0.0

                try:
                    score2_ind = self.patient_dict[patient_name2][part_name2]["induration"]
                except:
                    score2_ind = 0.0

                try:
                    score2_pasi = self.patient_dict[patient_name2]["pasi"]
                except:
                    continue

                score2 = [score2_area, score2_ery, score2_sca, score2_ind, score2_pasi]

                _img2 = Image.open(img2)

                try:
                    img2, box2 = self.valid_data_preprocessing(_img2, box2)
                except:
                    raise

                map_label2 = self.get_disease_map_label(box2)

                yield_img = [np.array([img1]), np.array([img2])]
                yield_label = [[score1], [score2], [[abs(elem[0] - elem[1]) for elem in zip(score1, score2)]]]

                [yield_label.append([elem]) for elem in map_label1]
                [yield_label.append([elem]) for elem in map_label2]

                # yield ({"input_a": img1, "input_b": img2},{"scoreA": score1, "scoreB": score2, "scoreSiam": abs(score1 - score2)})
                yield img1_basename, img2_basename, yield_img, yield_label
                # yield img1, img2, score1, score2, abs(score1-score2), map_label1[:], map_label2[:]

            else:
                break
