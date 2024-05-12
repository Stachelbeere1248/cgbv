import tkinter
from PIL import Image, ImageTk
import math
import click

# brightness of an RGB tuple, factors taken from
# https://de.mathworks.com/help/matlab/ref/rgb2gray.html
B = lambda r,g,b: 0.298936021293775 * r + 0.587043074451121 * g + 0.114020904255103 * b
G_x = [[-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]]
G_y = [[-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]]

def for_pixel(img: Image, border: (int, int, int, int), fn, args: tuple) -> Image:
    w,h = img.size
    out = img.copy()
    assert 0 <= border[0] <= w-1, f"left padding {border} is invalid"
    assert 0 <= border[1] <= h-1, f"top padding {border} is invalid"
    assert 0 <= border[2] <= w-1, f"right padding {border} is invalid"
    assert 0 <= border[3] <= h-1, f"bottom padding {border} is invalid"
    assert border[0] + border[2] < w
    assert border[1] + border[3] < h
    x_sel = range(border[0], w - border[2])
    y_sel = range(border[1], h - border[3])
    with click.progressbar(x_sel, label="iterating over image") as width:
        for x in width:
            for y in y_sel:
                pixel = fn(img, (x,y), args)
                out.putpixel((x,y), pixel)
    return out

def apply_kernel(img: Image, pos: (int, int), args: (list, int)) -> tuple:
    w,h = img.size
    x,y = pos
    kernel, kernel_size = args
    value = img.getpixel(pos)
    if isinstance(value, int):
        value = (value,)
    weighted_sum = tuple(0 for _ in value)
    for xoff in range(-kernel_size, kernel_size+1):
        for yoff in range(-kernel_size, kernel_size+1):
            factor = kernel[xoff][yoff]
            abs_x, abs_y = x+xoff, y+yoff
            off_value = value
            if 0 <= abs_x < img.width and 0 <= abs_y < img.height:
                off_value = img.getpixel((abs_x,abs_y))
                if isinstance(off_value, int):
                    off_value = (off_value,)
            weight = kernel[xoff+kernel_size][yoff+kernel_size]
            # handle unknown amount of color channels
            weighted_value = tuple(channel * factor for channel in off_value)
            # add channels element-wise to weighted_sum (thanks to chat-gpt for telling me about zip())
            weighted_sum = tuple(a + b for a,b in zip(weighted_sum, weighted_value))
    return tuple(int(channel) for channel in weighted_sum)

def scalar_mult(kernel,scalar) -> list:
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

def generate_blur(s: float) -> list:
    # I tried figuring out a similar formula myself but it wasn't perfect so I looked it up
    # Source: https://en.wikipedia.org/wiki/Gaussian_blur
    blur = lambda x,y,s: math.exp(-(x*x+y*y)/(2*s*s))

    radius = 3*math.ceil(s)
    size = 2*radius
    middle = radius
    normaliser = 2*math.pi*s*s
    kernel = list()
    for x in range(size+1):
        row = list()
        for y in range(size+1):
            val = blur(x-middle,y-middle,s)
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

def contrast_on_pixel(img: Image, pos: (int, int), args: (float, float)) -> tuple:
    factor, offset = args
    value = img.getpixel(pos)
    return tuple(max(0, min(
        int(((channel / 255.0 - offset) * factor + offset) * 255.0),
        255)) for channel in value)

def threshold_on_pixel(img: Image, pos: (int, int), arg: int) -> tuple:
    value = img.getpixel(pos)
    channels = len(value)
    if channels>=3:
        value = B(value[0],value[1],value[2])
    value = (value>=arg)*255
    return tuple(value for _ in range(channels))

def swirl_on_pixel(img: Image, pos: (int, int), args) -> tuple:
    w,h = img.size
    center, basef = args
    cx, cy = center
    x,y = pos
    dx, dy = x-cx, y-cy
    d = math.sqrt(dx*dx+dy*dy)
    s = math.pi * math.pow(basef,d)
    a = math.atan2(dy, dx)
    nx, ny = cx + math.cos(s + a) * d, cy + math.sin(s + a) * d
    nx, ny = max(nx,0), max(ny,0)
    nx, ny = min(nx, w-1), min(ny, h-1)
    return img.getpixel((nx,ny))

def sobel(img: Image, pos: (int, int), args):
    gray = args
    value = gray.getpixel(pos)
    args_x = (G_x, (validate_kernel(G_x)-1)//2)
    x_gradient, = apply_kernel(gray, pos, args_x)
    args_y = (G_y, (validate_kernel(G_y)-1)//2)
    y_gradient, = apply_kernel(gray, pos, args_y)
    gradient = int(math.sqrt(x_gradient*x_gradient + y_gradient*y_gradient))
    theta = math.atan2(y_gradient, x_gradient)
    theta /= math.pi
    theta = round(4*theta)
    theta = theta%4
    return (value, gradient, theta)

def supress_non_maximum(img: Image, pos: (int, int), args):
    _, gradient, angle = img.getpixel(pos)
    x,y = pos
    w,h = args
    pos1,pos2 = (x,y),(x,y)
    # I have absolutely no idea why it's shifted by 1 but this has the correct results I think
    match angle:
        case 1:
            pos1 = min(x+1,w-1),y
            pos2 = max(x-1,0),y
        case 2:
            pos1 = min(x+1,w-1),max(y-1,0)
            pos2 = max(x-1,0),min(y+1,h-1)
        case 3:
            pos1 = x,max(y-1,0)
            pos2 = x,min(y+1,h-1)
        case 0:
            pos1 = max(x-1,0),max(y-1,0)
            pos2 = min(x+1,w-1),min(y+1,h-1)
    _, side1, _ = img.getpixel(pos1)
    _, side2, _ = img.getpixel(pos2)
    if not (side1 <= gradient >= side2):
        gradient = 0
    return gradient,gradient,gradient

def double_threshold(img: Image, pos: (int, int), args):
    t1,t2 = args
    gradient = img.getpixel(pos)
    new_value = 0
    if gradient >= t1:
        new_value = 127
    if gradient >= t2:
        new_value = 255
    return new_value

def hysteresis(img: Image, pos: (int, int), args):
    value = img.getpixel(pos)
    w,h = args
    x,y = pos
    if value == 127:
        boolean = False
        for x_off in range (-1,2):
            for y_off in range(-1,2):
                x_off, y_off = x+x_off, y+y_off
                x_off, y_off = max(x_off,0), max(y_off,0)
                x_off, y_off = min(x_off, w-1), min(y_off, h-1)
                if img.getpixel((x_off,y_off)) == 255:
                    boolean = True
        if boolean:
            value = 0
        else:
            value = 255
    return value

class ImageContext:
    def __init__(self, filename):
        path = click.format_filename(filename)
        self.img = Image.open(path)

@click.group(chain=True)
@click.argument("filename", type=click.Path(exists = True), help="Path to the input image")
@click.pass_context
def process(ctx, filename):
    ctx.obj = ImageContext(filename)

@process.command()
@click.option("--value", "-v", type=int, default=127, help="The value above which pixels turn white")
@click.option("--padding", "-p", type=(int, int, int, int), default=(0,0,0,0), help="The padding on the left, top, right, and bottom")
@click.pass_context
def threshold(ctx, value: int, padding):
    ctx.obj.img = for_pixel(ctx.obj.img, padding, threshold_on_pixel, value)

@process.command()
@click.option("--factor", "-f", type=float, default=1, help="The factor by which to scale brightness values")
@click.option("--offset", "-o", type=float, default=0, help="The relative center for scaling the brightness")
@click.option("--padding", "-p", type=(int, int, int, int), default=(0,0,0,0), help="The padding on the left, top, right, and bottom")
@click.pass_context
def contrast(ctx, factor, offset, padding):
    args = factor, offset
    ctx.obj.img = for_pixel(ctx.obj.img, padding, contrast_on_pixel, args)

@process.command()
@click.option("--factor", "-f", type=float, default=1, help="The factor by which to increase pixel values")
@click.option("--padding", "-p", type=(int, int, int, int), default=(0,0,0,0), help="The padding on the left, top, right, and bottom")
@click.pass_context
def brightness(ctx, factor, padding):
    kernel = [[factor]]
    size = (validate_kernel(kernel)-1)//2
    args = kernel, size
    ctx.obj.img = for_pixel(ctx.obj.img, padding, apply_kernel, args)

@process.command()
@click.option("--deviation", "-s", type=float, default=1, help="The standard deviation of the gaussian blur. High values might cause significant computation time on larger images")
@click.option("--padding", "-p", type=(int, int, int, int), default=(0,0,0,0), help="The padding on the left, top, right, and bottom")
@click.pass_context
def blur(ctx, deviation, padding):
    kernel = generate_blur(deviation)
    size = (validate_kernel(kernel)-1)//2
    args = kernel, size
    ctx.obj.img = for_pixel(ctx.obj.img, padding, apply_kernel, args)

@process.command()
@click.option("--deviation", "-s", type=float, default=1, help="The standard deviation of the gaussian blur. High values might cause significant computation time on larger images")
@click.option("--amount", "-a", type=float, default=1, "Amount of highlighting outstanding pixels")
@click.option("--padding", "-p", type=(int, int, int, int), default=(0,0,0,0), help="The padding on the left, top, right, and bottom")
@click.pass_context
def sharpen(ctx, deviation, amount, padding):
    kernel = generate_blur(deviation)
    kernel = convert_to_usm(kernel, amount)
    size = (validate_kernel(kernel)-1)//2
    args = kernel, size
    ctx.obj.img = for_pixel(ctx.obj.img, padding, apply_kernel, args)

@process.command()
@click.option("--base", "-b", type=float, default=0.95, help="Base for exponential decay. Larger values increase the swirl effect's radius.")
@click.option("--padding", "-p", type=(int, int, int, int), default=(0,0,0,0), help="The padding on the left, top, right, and bottom")
@click.pass_context
def swirl(ctx, base, padding):
    w,h = ctx.obj.img.size
    center = (w//2, h//2)
    args = (center, base, radius)
    ctx.obj.img = for_pixel(ctx.obj.img, padding, swirl_on_pixel, args)

@process.command()
@click.option("--threshold", "-t", type=(int, int), default=(20,55), help="Values for considering weak and strong edges.")
@click.pass_context
def edge(ctx, threshold):
    padding = (0,0,0,0)
    img = ctx.obj.img.convert("RGB")
    kernel = generate_blur(1.5)
    size = (validate_kernel(kernel)-1)//2
    args = kernel, size
    img = for_pixel(img, padding, apply_kernel, args)
    gray = img.convert("L")
    data = for_pixel(img, padding, sobel, gray)
    data = for_pixel(data, padding, supress_non_maximum, data.size)
    data = data.convert("L")
    data = for_pixel(data, padding, double_threshold, threshold)
    img = for_pixel(data, padding, hysteresis, data.size)
    ctx.obj.img = img

@process.command()
@click.pass_context
def show(ctx):
    img = ctx.obj.img
    w,h = img.size
    window = tkinter.Tk(className="filtertool")
    canvas = tkinter.Canvas(window, width=w, height=h)
    canvas.pack()
    image_tk = ImageTk.PhotoImage(img)
    canvas.create_image(w//2, h//2, image=image_tk)
    tkinter.mainloop()

@process.command
@click.pass_context
def save(ctx):
    ctx.obj.img.save("output.png")

if __name__ == '__main__':
    process(obj={})


