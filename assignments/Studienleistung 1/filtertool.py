from PIL import Image
import math
i_input = Image.open("input.png")

# brightness of an RGB tuple, factors taken from
# https://de.mathworks.com/help/matlab/ref/rgb2gray.html
B = lambda r,g,b: 0.298936021293775 * r + 0.587043074451121 * g + 114020904255103 * b
# identity
I = [[1]]
# I tried figuring out a similar formula myself but it wasn't perfect so I looked it up
# Source: https://en.wikipedia.org/wiki/Gaussian_blur
blur = lambda x,y,s: math.exp(-(x*x+y*y)/(2*s*s))

def apply_kernel(img: Image, kernel: list) -> Image:
    img_applied = img.copy()
    kernel_size = (validate_kernel(kernel)-1)//2
    num_channels = len(img.getpixel((0,0)))

    # iterate over image pixels
    for x in range(img_applied.width):
        for y in range(img_applied.height):
            # apply kernel on pixel
            weighted_sum = tuple(0 for _ in range(num_channels))
            for xoff in range(-kernel_size, kernel_size+1):
                for yoff in range(-kernel_size, kernel_size+1):
                    factor = kernel[xoff][yoff]
                    abs_x, abs_y = x+xoff, y+yoff
                    value = img.getpixel((x,y))
                    if 0 <= abs_x < img.width and 0 <= abs_y < img.height:
                        value = img.getpixel((abs_x,abs_y))
                    weight = kernel[xoff+kernel_size][yoff+kernel_size]
                    # handle unknown amount of color channels
                    weighted_value = tuple(channel * factor for channel in value)
                    # add channels element-wise to weighted_sum (thanks to chat-gpt for telling me about zip())
                    weighted_sum = tuple(a + b for a,b in zip(weighted_sum, weighted_value))
            weighted_sum = tuple(int(channel) for channel in weighted_sum)
            img_applied.putpixel((x,y), weighted_sum)
    return img_applied

def scalar_mult(kernel,scalar):
    k1 = list()
    for row in kernel:
        row1 = list()
        for value in row:
            row1.append(value*scalar)
        k1.append(row1)
    return k1

def kernel_sum(kernel:list) -> float:
    sum = 0.0
    for row in kernel:
        for val in row:
            sum += val
    return sum

# Assert that the kernel is an odd square and return its width
def validate_kernel(kernel: list) -> int:
    len_k = len(kernel)
    assert len_k-1 >= 0, f"size {len_k} is too small for a kernel"
    assert len_k%2 == 1, f"please only use odd kernel sizes, got: {len_k}"
    for row in kernel:
        len_r = len(row)
        assert len_r == len_k, f"this kernel row seems deformed: {row}, expected a row of length {len_k}"
    return len_k

def generate_blur(fn, radius: int, s: float) -> list:
    size = 2*radius
    middle = radius
    normaliser = 2*math.pi*s*s
    kernel = list()
    for x in range(size+1):
        row = list()
        for y in range(size+1):
            val = fn(x-middle,y-middle,s)
            row.append(val)
        kernel.append(row)
    kernel = [[value/normaliser for value in row] for row in kernel]
    return kernel

# I think this should generate an unsharpening mask for a given blur
# Source: https://en.wikipedia.org/wiki/Unsharp_masking
def convert_to_usm(blur: list, amount: float):
    k1 = scalar_mult(blur, -amount)
    leng = validate_kernel(k1)
    k1[leng//2][leng//2] += amount+1
    return k1

def contrast(img: Image, factor):
    out = img.copy()
    w,h = img.size
    def contrast_on_pixel(value):
        return tuple(max(0, min(
            int(((channel / 255.0 - 0.5) * factor + 0.5) * 255.0),
            255)) for channel in value)

    for x in range(w):
        for y in range(h):
            value = img.getpixel((x,y))
            out.putpixel((x,y),contrast_on_pixel(value))
    return out

blur_kernel = generate_blur(blur, 3, 1)
usm1 = convert_to_usm(blur_kernel, 0.5)
contrast(i_input, 1.5).save("output.png")
