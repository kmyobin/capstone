import matplotlib.pyplot as plt
from PIL import Image

image=Image.open('eyebrow_synthesis/results/eyebrow_remove/test_latest/images/pair_fake_B.png')

#image.save('demo_result_dataset/pair.png')
image.save('output/irene_remove.png')

plt.imshow(image)
plt.show()