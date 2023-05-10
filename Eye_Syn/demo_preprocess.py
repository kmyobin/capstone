from PIL import Image
import cv2

# 이미지 두개 만드는 작업

image1 = Image.open('output/irene_remove.png')
image2 = Image.open('output/irene_remove.png')

result = Image.new('RGB', (image1.size[0] + image2.size[0], image1.size[1]))
result.paste(im=image1, box=(0, 0))
result.paste(im=image2, box=(image1.size[0], 0))

#result.save('demo_result_dataset/pair.jpg')
result.save('eyebrow_synthesis/datasets/test_data/test/pair.jpg')