import os
import pandas as pd
import shutil

# Folder containing all the xml metadata files:
xmlPath = "C:/Users/seana/Homework/L3D/RijksData/xml/"
jpgPath = "C:/Users/seana/Homework/L3D/RijksData/jpg/"

materials = [
'papier',
'zilver',
'faience',
'porselein',
'hout',
'brons',
'glas (materiaal)',
'perkament',
'geprepareerd papier',
'fotopapier',
'ijzer',
'Japans papier',
'ivoor',
'Oosters papier',
'eikenhout']

def extract_info(xmlFile):
    with open(xmlFile, 'r', encoding="utf-8") as f:
        xmlStr = f.read()
    
    materials = []
    creators = []

    # Define search strings for each category
    search_patterns = {
        "material": ("<dc:format>materiaal: ", materials),
        "creator": ("<dc:creator>", creators)
    }

    for key, (matchStr, collection) in search_patterns.items():
        begin = xmlStr.find(matchStr)
        while begin != -1:
            end = xmlStr.find("<", begin + len(matchStr))
            value = xmlStr[begin + len(matchStr):end]
            collection.append(value)
            begin = xmlStr.find(matchStr, end)

    return materials, creators

def create_df():
    img_data = []
    for file in os.scandir(xmlPath):
        if file.is_file():
            materials_found, creators_found = extract_info(file.path)
            if len(materials_found) == 1 and materials_found[0] in materials:
                # Join 'creators' lists into comma-separated strings
                creators_str = "|".join(creators_found).strip()

                # Append a new row to the DataFrame
                img_data.append({
                    "filename": file.name.replace(".xml", ".jpg"),
                    "material": materials_found[0],
                    "creators": creators_str
                })

    return img_data

data_file = 'data_annotations/dataset_info.csv'
if os.path.exists(data_file):
    full_df = pd.read_csv(data_file)
else:
    img_data = create_df()
    full_df = pd.DataFrame(img_data).sample(frac=1)
    full_df.to_csv(data_file, index=False)

def createMaterialHist(mat_df):
    mat_hist = {}
    for index, row in mat_df.iterrows():
        if row["material"] in mat_hist:
            mat_hist[row["material"]] += 1
        else:
            mat_hist[row["material"]] = 1

    # Convert to sorted list:
    hist_list = [[material, mat_hist[material]] for material in mat_hist]
    hist_list.sort(key = lambda x: x[1], reverse = True)

    return hist_list

def createComboHist(combo_df):
    combo_hist = {}
    for index, info in combo_df.iterrows():
        combined_info = tuple([info["material"],info["creators"]])
        if combined_info in combo_hist:
            combo_hist[combined_info] += 1
        else:
            combo_hist[combined_info] = 1

    # Convert to sorted list:
    hist_list = [[info, combo_hist[info]] for info in combo_hist]
    hist_list.sort(key = lambda x: x[1], reverse = True)

    return hist_list

def printHist(hist, file):
    total = 0
    for row in hist:
        total += row[1]
    with open(file, 'a') as f:
        for row in hist:
            f.writelines(f"{row[0]}, {row[1]}, {(100 * row[1] / total):>0.1f}%\n")

full_mat_hist = createMaterialHist(full_df)
full_combo_hist = createComboHist(full_df)
# printHist(full_mat_hist, 'data_annotations/full_histogram.txt')
# printHist(full_combo_hist, 'data_annotations/full_combo_histogram.txt')

def splitData():
    # all: non-overlapping creators
    # train set: equal number of instances for all classes
    # val set: 1 instance of each class
    # test set: random sample
    train_list = []
    train_material_instances = 80
    train_material_counter = {key: 0 for key in materials}
    train_combo_counter = {}
    val_list = []
    val_material_instances = 10
    val_material_counter = {key: 0 for key in materials}
    val_combo_counter = {}
    test_list = []
    test_material_instances = 60
    test_material_counter = {key: 0 for key in materials}
    test_combo_counter = {}

    for index, info in full_df.iterrows():
        material = info["material"]
        creators = info["creators"]
        img = info["filename"]
        combined_info = tuple([material,creators])

        class_size = 0
        for combo_count in full_combo_hist:
            if combo_count[0] == combined_info:
                class_size = combo_count[1]
                break

        if val_material_counter[material] < val_material_instances and class_size < 10:
            if combined_info not in val_combo_counter:
                val_combo_counter[combined_info] = 0
            val_list.append([img, material])
            val_material_counter[material] += 1
            val_combo_counter[combined_info] += 1
            continue

        # fill train set next as long as same material/artist not already in validation set
        elif combined_info not in val_combo_counter:
            if train_material_counter[material] < train_material_instances:
                if combined_info not in train_combo_counter:
                    train_combo_counter[combined_info] = 0
                train_list.append([img, material])
                train_material_counter[material] += 1
                train_combo_counter[combined_info] += 1
                continue

            # fill test set last with whatever material/artist is not already in train or validation set
            elif test_material_counter[material] < test_material_instances and combined_info not in train_combo_counter:
                if combined_info not in test_combo_counter:
                    test_combo_counter[combined_info] = 0
                test_list.append([img, material])
                test_material_counter[material] += 1
                test_combo_counter[combined_info] += 1
                continue

    for material in materials:
        if train_material_counter[material] < train_material_instances:
            print(f"incomplete {material} training set of size {train_material_counter[material]}!!!")

    train_df = pd.DataFrame(train_list, columns=['filename', 'material'])
    train_df.to_csv("data_annotations/train_df.csv", index=False)
    train_mat_hist = [[material, train_material_counter[material]] for material in train_material_counter]
    train_mat_hist.sort(key = lambda x: x[1], reverse = True)
    printHist(train_mat_hist, 'data_annotations/train_histogram.txt')
    train_combo_hist = [[combo, train_combo_counter[combo]] for combo in train_combo_counter]
    train_combo_hist.sort(key = lambda x: x[1], reverse = True)
    printHist(train_combo_hist, 'data_annotations/train_combo_histogram.txt')

    val_df = pd.DataFrame(val_list, columns=['filename', 'material'])
    val_df.to_csv("data_annotations/val_df.csv", index=False)
    val_mat_hist = [[material, val_material_counter[material]] for material in val_material_counter]
    val_mat_hist.sort(key = lambda x: x[1], reverse = True)
    printHist(val_mat_hist, 'data_annotations/val_histogram.txt')
    val_combo_hist = [[combo, val_combo_counter[combo]] for combo in val_combo_counter]
    val_combo_hist.sort(key = lambda x: x[1], reverse = True)
    printHist(val_combo_hist, 'data_annotations/val_combo_histogram.txt')

    test_df = pd.DataFrame(test_list, columns=['filename', 'material'])
    test_df.to_csv("data_annotations/test_df.csv", index=False)
    test_mat_hist = [[material, test_material_counter[material]] for material in test_material_counter]
    test_mat_hist.sort(key = lambda x: x[1], reverse = True)
    printHist(test_mat_hist, 'data_annotations/test_histogram.txt')
    test_combo_hist = [[combo, test_combo_counter[combo]] for combo in test_combo_counter]
    test_combo_hist.sort(key = lambda x: x[1], reverse = True)
    printHist(test_combo_hist, 'data_annotations/test_combo_histogram.txt')

    return train_df, val_df, test_df

train_file = 'data_annotations/train_df.csv'
val_file = 'data_annotations/val_df.csv'
test_file = 'data_annotations/test_df.csv'
if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
else:
    train_df, val_df, test_df = splitData()

# merged_df = pd.merge(train_df, val_df, how='inner')
# overlap_df = pd.merge(merged_df, test_df, how='inner')
# if not overlap_df.empty:
#     print("There are overlapping rows.")
#     print(overlap_df)
# else:
#     print("No overlapping rows found.")

train_xmlPath = "C:/Users/seana/Homework/L3D/RijksData/train_xml/"
train_jpgPath = "C:/Users/seana/Homework/L3D/RijksData/train_jpg/"
val_xmlPath = "C:/Users/seana/Homework/L3D/RijksData/val_xml/"
val_jpgPath = "C:/Users/seana/Homework/L3D/RijksData/val_jpg/"
test_xmlPath = "C:/Users/seana/Homework/L3D/RijksData/test_xml/"
test_jpgPath = "C:/Users/seana/Homework/L3D/RijksData/test_jpg/"
def moveSubsetFiles():
    for index, train_info in train_df.iterrows():
        jpg_filename = train_info["filename"]
        src_jpg_path = os.path.join(jpgPath, jpg_filename)
        dest_jpg_path = os.path.join(train_jpgPath, jpg_filename)
        shutil.move(src_jpg_path, dest_jpg_path)
        xml_filename = jpg_filename.replace(".jpg", ".xml")
        src_xml_path = os.path.join(xmlPath, xml_filename)
        dest_xml_path = os.path.join(train_xmlPath, xml_filename)
        shutil.move(src_xml_path, dest_xml_path)

    for index, val_info in val_df.iterrows():
        jpg_filename = val_info["filename"]
        src_jpg_path = os.path.join(jpgPath, jpg_filename)
        dest_jpg_path = os.path.join(val_jpgPath, jpg_filename)
        shutil.move(src_jpg_path, dest_jpg_path)
        xml_filename = jpg_filename.replace(".jpg", ".xml")
        src_xml_path = os.path.join(xmlPath, xml_filename)
        dest_xml_path = os.path.join(val_xmlPath, xml_filename)
        shutil.move(src_xml_path, dest_xml_path)
    
    for index, test_info in test_df.iterrows():
        jpg_filename = test_info["filename"]
        src_jpg_path = os.path.join(jpgPath, jpg_filename)
        dest_jpg_path = os.path.join(test_jpgPath, jpg_filename)
        shutil.move(src_jpg_path, dest_jpg_path)
        xml_filename = jpg_filename.replace(".jpg", ".xml")
        src_xml_path = os.path.join(xmlPath, xml_filename)
        dest_xml_path = os.path.join(test_xmlPath, xml_filename)
        shutil.move(src_xml_path, dest_xml_path)

moveSubsetFiles()