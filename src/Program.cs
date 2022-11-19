using System.Drawing;
using WarpTest;
using Emgu.CV;
using Emgu.CV.CvEnum;

var options = CmdLine.ProcessArgs(args);
if (!CmdLine.ValidateOptions(options)) return;

// Translated example of https://stackoverflow.com/questions/12017790/warp-image-to-appear-in-cylindrical-projection

Mat imgsrc;
using (var img = CvInvoke.Imread(options.ImagePath))
{
    if (options.BorderSize.Width > 0 || options.BorderSize.Height > 0)
    {
        // Add a border around the source image.
        int bw = options.BorderSize.Width;
        int bh = options.BorderSize.Height;
        imgsrc = new Mat(new Size(bw * 2, bh * 2), Emgu.CV.CvEnum.DepthType.Cv8U, 3);
        CvInvoke.CopyMakeBorder(img, imgsrc, bh, bh, bw, bw, Emgu.CV.CvEnum.BorderType.Constant);
    }
    else
    {
        // No border specified.
        imgsrc = img.Clone();
    }
}
// Display the source image in a window.
CvInvoke.Imshow("original", imgsrc);

//rows=height, cols=width eg (y,x), first dim=x, second dim=y for uv map
using var mapX = new Matrix<float>(imgsrc.Height,imgsrc.Width);
using var mapY = new Matrix<float>(imgsrc.Height, imgsrc.Width);

// Are we mapping along the x- or y-axis? Set our mapping function.
Func<PointF, int, int, float, PointF> map = options.HemisphereMap ? MapHemisphere : options.MapXAxis ? MapCylinderX : MapCylinderY;

// Are we mapping a convex or concave surface?
float zDir = options.Convex ? -1f : 1f;

// Calculate the x & y values.
for (int y = 0; y < imgsrc.Height; y++)
{
    for (int x = 0; x < imgsrc.Width; x++)
    {
        var pointOriginal = new PointF(x, y);
        var pointCylinder=map(pointOriginal, imgsrc.Width, imgsrc.Height, zDir);

        mapX[y, x] = pointCylinder.X;
        mapY[y, x] = pointCylinder.Y;
    }
}

using var imgdest = imgsrc.Clone();

// Remap the source image using our calculated coordinates.
CvInvoke.Remap(imgsrc, imgdest, mapX, mapY, Emgu.CV.CvEnum.Inter.Linear);

// Display the re-mapped image in a window.
CvInvoke.Imshow("remapped", imgdest);

CvInvoke.WaitKey();

static PointF MapCylinderY(PointF point, int w, int h, float zDir)
{
    // Center the point at (0,0)
    var pc = new PointF(point.X - w / 2, point.Y - h / 2);

    // Free parameters that can be adjust for interesting effects.
    float f = w * zDir;
    //float f = w/2;
    //float f = 2*w;
    float r = w / 2;
    //float r = w/2;
    //float r = 2 * w; //cylinder diameter is smaller than the width of the image

    float omega = w / 2;
    float z0 = f - (float)Math.Sqrt(f * f - omega * omega);

    float zc = (float)((2.0 * z0 + Math.Sqrt(4.0 * z0 * z0 - 4.0 * (pc.X * pc.X / (f * f) + 1.0) * (z0 * z0 - r * r))) / (2.0 * (pc.X * pc.X / (f * f) + 1.0)));
    // The calculation below skews the image.
    // float zc = (2 * z0 + (float)Math.Sqrt(4 * z0 * z0 - 4 * (pc.X * pc.Y / (f * f) + 1) * (z0 * z0 - r * r))) / (2 * (pc.X * pc.X / (f * f) + 1));
    var final_point = new PointF(pc.X * zc / f, pc.Y * zc / f);
    final_point.X += w / 2;
    final_point.Y += h / 2;

    return final_point;
}

// Copied the MapCylinderY(..) function and changed a few lines rather than
// having one function with a bunch of if (mapX) ... else ...tests inside the loop.
static PointF MapCylinderX(PointF point, int w, int h, float zDir)
{
    // Center the point at (0,0)
    var pc = new PointF(point.X - w / 2, point.Y - h / 2);

    // Free parameters that can be adjust for interesting effects.
    float f = h * zDir;
    //float f = w/2;
    //float f = 2*w;
    float r = h / 2;
    //float r = w/2;
    //float r = 2 * w; //cylinder diameter is smaller than the width of the image

    float omega = h / 2;
    float z0 = f - (float)Math.Sqrt(f * f - omega * omega);

    float zc = (float)((2.0 * z0 + Math.Sqrt(4.0 * z0 * z0 - 4.0 * (pc.Y * pc.Y / (f * f) + 1.0) * (z0 * z0 - r * r))) / (2.0 * (pc.Y * pc.Y / (f * f) + 1.0)));
    // The calculation below skews the image.
    // float zc = (2 * z0 + (float)Math.Sqrt(4 * z0 * z0 - 4 * (pc.X * pc.Y / (f * f) + 1) * (z0 * z0 - r * r))) / (2 * (pc.X * pc.X / (f * f) + 1));
    var final_point = new PointF(pc.X * zc / f, pc.Y * zc / f);
    final_point.X += w / 2;
    final_point.Y += h / 2;

    return final_point;
}

static PointF MapHemisphere(PointF point, int w, int h, float zDir)
{
    // Center the point at (0,0)
    var pc = new PointF(point.X - w / 2, point.Y - h / 2);

    // Free parameters that can be adjust for interesting effects.
    float fx = w * zDir;
    float rx = w / 2;

    float omegax = w / 2;
    float zx0 = fx - (float)Math.Sqrt(fx * fx - omegax * omegax);

    float fy = w * zDir;
    float ry = w / 2;

    float omegay = w / 2;
    float zy0 = fy - (float)Math.Sqrt(fy * fy - omegay * omegay);
    //x^2 + (z - zx0)^2 - rx^2 = 0
    //y^2 + (z - zy0)^2 - ry^2 = 0
    //===
    //((pcX^2 / fx^2) + (pcY^2 / fy^2) + 2) * z^2 - 2 * (zx0 + zy0) * z + (zx0 * zx0) + (zy0 * zy0) - (rx * rx) - (ry * ry)
    float a = (float)(((pc.X * pc.X) / (fx * fx)) + ((pc.Y * pc.Y) / (fy * fy)) + 2.0);
    float b = (float)(-2.0 * (zx0 + zy0));
    float c = (float)((zx0 * zx0) + (zy0 * zy0) - (rx * rx) - (ry * ry));
    //Quadratic (-b + sqrt(b^2 - 4ac)) / 2a
    float zc = (float)(((-1.0 * b) + Math.Sqrt((b * b) - (4.0 * (a) * (c)))) / (2.0 * (a)));
    var final_point = new PointF(pc.X * zc / fx, pc.Y * zc / fy);
    final_point.X += w / 2;
    final_point.Y += h / 2;

    return final_point;
}