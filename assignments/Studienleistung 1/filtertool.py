from PIL import Image
i_input = Image.open("input.png")

def apply_kernel(img: Image, kernel: list) -> Image:
    img_applied = img.copy()
    kernel_size = (len(kernel)-1)//2
    num_channels = len(img.getpixel((0,0)))

    # get normaliser for kernel
    normaliser = 0
    for row in kernel:
        for factor in row:
            normaliser += factor

    # iterate over image pixels
    for x in range(img_applied.width):
        for y in range(img_applied.height):

            # apply kernel on pixel
            weighted_sum = tuple(0 for _ in range(num_channels))
            for xoff in range(-kernel_size, kernel_size+1):
                for yoff in range(-kernel_size, kernel_size+1):
                    abs_x, abs_y = x+xoff, y+yoff
                    # bad solution, but it works
                    if abs_x not in range(img.width):
                        abs_x = 0
                    if abs_y not in range(img.height):
                        abs_y = 0
                    value = img.getpixel((abs_x, abs_y))
                    weight = kernel[xoff+kernel_size][yoff+kernel_size]
                    # handle unknown amount of color channels
                    weighted_value = tuple(channel * factor for channel in value)
                    # add channels element-wise to weighted_sum (thanks to chat-gpt for telling me about zip())
                    weighted_sum = tuple(a + b for a,b in zip(weighted_sum, weighted_value))
            
            # kernel probably wasn't normalised
            new_value = tuple(channel//normaliser for channel in weighted_sum)
            img_applied.putpixel((x,y), new_value)
    return img_applied

test_kernel = [[1,2,1],[2,4,2],[1,2,1]]

apply_kernel(i_input, test_kernel).save("output.png")
