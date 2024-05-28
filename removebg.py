import cv2
import numpy as np
import imageio
from tqdm import tqdm
seqname='giraffe2'
img_path = f'datasource/{seqname}/imgs/'
mask_path = f'datasource/{seqname}/masks/'
output_path = f'datasource/{seqname}/'
def remove_background(image_path, mask_path, output_path):
    # Read the image and mask
    # print(image_path, mask_path, output_path)
    image = cv2.imread(image_path).astype(np.float32)/255.
    mask = cv2.imread(mask_path)[:,:,:1].astype(np.float32)/255.

    # Apply the mask to the image
    result = image*mask+(1-mask)

    # Save the result to the output path
    cv2.imwrite(output_path, result*255.)
    return result

# Example usage
n=50
frames = {'camel':(0,25),'bailang':(8,31),'zongxiong':(0,77),'snail':(0,94),'penguin_n':(0,37),'littlelion':(5,43)}
# a,b = frames[seqname]
imgs = []
for i in tqdm(range(1,n)):
    number = '%05d.jpg'%(i)
    result = remove_background(img_path+number, mask_path+'%05d.jpg'%(i), output_path+number)
    imgs.append(result[:,:,::-1])

# imgs = (imgs*5)[:80]
# imageio.mimsave('%s%s'%(output_path,'ref.mp4'), imgs, fps=10)