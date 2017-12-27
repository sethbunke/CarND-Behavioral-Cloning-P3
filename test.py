import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


print('test')


def adjust_angle_per_camera(features, camera):
    
   
    correction = 0.25
    if camera == 'left':
        features[0] = features[0] + correction
    elif camera == 'right':
        features[0] = features[0] - correction
    return features

data_version = '1'
csv_file_name = '../simulator/data1/driving_log.csv'
img_file_path = '../simulator/data1/IMG/'

lines = []

with open(csv_file_name) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

def get_images_and_measurements(inner_line, steering_correction):
    inner_images = []
    inner_measurements = []

    for index in range(0,3):                
        steering_angle = float(inner_line[3])        
        #left image
        if index == 1:            
            steering_angle = steering_angle + steering_correction
        #right image
        elif index == 2:            
            steering_angle = steering_angle - steering_correction
        #else - center

        source_path = inner_line[index]
        filename = source_path.split('/')[-1]    
        current_path = img_file_path + filename
        image = cv2.imread(current_path)
        inner_images.append(image)
        measurement = steering_angle
        inner_measurements.append(measurement)

        inner_images.append(cv2.flip(image, 1)) #flip image
        inner_measurements.append(measurement * -1.0) #invert value

    return (inner_images, inner_measurements)

images = []
measurements = []

for line in lines:
    imgs, meas = get_images_and_measurements(line, 0.2)
    images.extend(imgs)
    measurements.extend(meas)

print(len(lines))
print(len(images))
print(len(measurements))


print('test')



    
    


   
    



# steering_correction = 0.2

# images = []
# measurements = []
# for line in lines:
    
#     steering_angle = float(line[3])
#     #center
#     source_path = line[0]
#     filename = source_path.split('/')[-1]    
#     current_path = img_file_path + filename
#     image = cv2.imread(current_path)
#     images.append(image)
#     measurement = steering_angle
#     measurements.append(measurement)

#     #left
#     source_path = line[1]
#     filename = source_path.split('/')[-1]    
#     current_path = img_file_path + filename
#     image = cv2.imread(current_path)
#     images.append(image)
#     measurement = steering_angle + steering_correction
#     measurements.append(measurement)

#     #right
#     source_path = line[2]
#     filename = source_path.split('/')[-1]    
#     current_path = img_file_path + filename
#     image = cv2.imread(current_path)
#     images.append(image)
#     measurement = steering_angle - steering_correction
#     measurements.append(measurement)






    







# [
# [[  462.55078125   161.32762146]]

#  [[  580.14978027   169.77209473]]

#  [[  687.98144531   183.75178528]]

#  [[  782.03601074   201.73049927]]

#  [[  861.84204102   220.25390625]]

#  [[  929.06414795   238.20245361]]

#  [[  984.19836426   255.16371155]]

#  [[ 1030.12451172   269.85070801]]

#  [[  456.59567261   274.07598877]]

#  [[  578.60742188   277.9262085 ]]

#  [[  690.21331787   287.43081665]]

#  [[  786.32885742   298.40231323]]

#  [[  866.9274292    310.83807373]]

#  [[  934.28497314   322.10079956]]

#  [[  989.39630127   333.20043945]]

#  [[ 1034.71533203   343.46920776]]

#  [[  454.92950439   397.90582275]]

#  [[  578.51080322   396.73190308]]

#  [[  691.67156982   398.89361572]]

#  [[  788.3223877    402.30404663]]

#  [[  868.85638428   405.93081665]]

#  [[  935.99414062   410.52624512]]

#  [[  990.74536133   414.49304199]]

#  [[ 1036.3515625    418.31710815]]

#  [[  458.96832275   524.45031738]]

#  [[  581.09515381   518.8338623 ]]

#  [[  692.28277588   513.17431641]]

#  [[  788.34008789   507.58029175]]

#  [[  867.75579834   502.92584229]]

#  [[  934.73413086   499.42382812]]

#  [[  989.39294434   495.57516479]]

#  [[ 1033.84179688   493.74246216]]

#  [[  467.61642456   644.09106445]]

#  [[  584.69696045   634.26733398]]

#  [[  692.28149414   621.68347168]]

#  [[  785.57080078   609.19256592]]

#  [[  864.2477417    596.84350586]]

#  [[  930.04327393   586.24420166]]

#  [[  983.95965576   575.96130371]]

#  [[ 1028.64526367   567.11663818]]

#  [[  478.81427002   748.91790771]]

#  [[  589.27954102   735.66900635]]

#  [[  690.95855713   719.79718018]]

#  [[  780.51525879   701.58233643]]

#  [[  857.66125488   684.11499023]]

#  [[  922.06231689   667.00042725]]

#  [[  976.35668945   651.64776611]]

#  [[ 1021.08343506   637.51074219]]
#  ]