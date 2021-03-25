import imageio
import glob

# get image files
site = 'kenx'
img_files = sorted(glob.glob(f'*{site}.png'))

#with imageio.get_writer(f'{site}.gif', mode='I', duration=0.15) as writer:
with imageio.get_writer(f'{site}.mp4', fps=2) as writer:
    #for pi in parr:
    for imf in img_files:
        image = imageio.imread(imf)
        writer.append_data(image)
