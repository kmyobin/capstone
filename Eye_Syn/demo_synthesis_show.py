import matplotlib.pyplot as plt
from PIL import Image

image=Image.open('eyebrow_synthesis/results/eyebrow_synthesis/test_latest/images/pair_fake_B.png')

#image.save('demo_result_dataset/pair.png')
image.save('output/irene_synthesis.png')

plt.imshow(image)
plt.show()