namespace WarpTest;

using System;
using System.Drawing;

record RemapOptions(string ImagePath, Size BorderSize, bool MapXAxis, bool Convex, bool ShowHelp, bool HemisphereMap);

static class CmdLine
{
    public const string DefaultImagePath = @".\images\grid.jpg";
    public const int DefaultBorderSize = 0;

    public static RemapOptions ProcessArgs(string[] args)
    {
        string? imgPath = DefaultImagePath;
        int borderWidth = DefaultBorderSize;
        int borderHeight = DefaultBorderSize;
        bool mapX = false;
        bool convex = false;
        bool showHelp = false;
        bool hemisphereMap = false;

        foreach (string arg in args)
        {
            if (!arg.StartsWith("-"))
            {
                imgPath = arg;
            }
            else if (arg.Equals("-x", StringComparison.OrdinalIgnoreCase) || arg.Equals("--mapx", StringComparison.OrdinalIgnoreCase))
            {
                mapX = true;
            }
            else if (arg.Equals("-v", StringComparison.OrdinalIgnoreCase) || arg.Equals("--convex", StringComparison.OrdinalIgnoreCase))
            {
                convex = true;
            }
            else if (arg == "-?" || arg.Equals("-h", StringComparison.OrdinalIgnoreCase) || arg.Equals("--help", StringComparison.OrdinalIgnoreCase))
            {
                showHelp = true;
                break;
            }
            else if (arg.StartsWith("-bx=", StringComparison.OrdinalIgnoreCase) || arg.StartsWith("-bx:", StringComparison.OrdinalIgnoreCase))
            {
                // left & right border size
                if (int.TryParse(arg.AsSpan(4), out var i))
                {
                    borderWidth = i;
                }
            }
            else if (arg.StartsWith("-by=", StringComparison.OrdinalIgnoreCase) || arg.StartsWith("-by:", StringComparison.OrdinalIgnoreCase))
            {
                // top & bottom border size
                if (int.TryParse(arg.AsSpan(4), out var i))
                {
                    borderHeight = i;
                }
            }
            else if (arg.StartsWith("-b=", StringComparison.OrdinalIgnoreCase) || arg.StartsWith("-b:", StringComparison.OrdinalIgnoreCase))
            {
                // all border size
                if (int.TryParse(arg.AsSpan(3), out var i))
                {
                    borderWidth = i;
                    borderHeight = i;
                }
            }
            else if (arg.StartsWith("-h", StringComparison.OrdinalIgnoreCase) || arg.StartsWith("--hemisphere", StringComparison.OrdinalIgnoreCase))
            {
                hemisphereMap= true;
            }
            else
            {
                Console.WriteLine($"Unknown option \"{arg}\"");
            }
        }

        return new(imgPath, new Size(borderWidth, borderHeight), mapX, convex, showHelp, hemisphereMap);
    }

    public static bool ValidateOptions(RemapOptions options)
    {
        if (options.ShowHelp)
        {
            ShowHelp();
            return false;
        }

        if (!File.Exists(options.ImagePath))
        {
            Console.WriteLine($"Error! No image file found at \"{options.ImagePath}\"");
            return false;
        }

        if ((options.BorderSize.Width * options.BorderSize.Height) < 0)
        {
            Console.WriteLine("Error! Border sizes cannot be negative.");
        }

        return true;
    }

    public static void ShowHelp()
    {
        Console.WriteLine("Cylinder Remap Example");
        Console.WriteLine("\nUsage: dotnet run -- [path-to-source-image] {options}");
        Console.WriteLine();
        Console.WriteLine("options:");
        Console.WriteLine("{0,-23} {1}", "  -? | -h | --help", "Show command line help.");
        Console.WriteLine("{0,-23} {1}", "  -x | --mapx", "Perform mapping along the x-axis (default is y-axis).");
        Console.WriteLine("{0,-23} {1}", "  -v | --convex", "Perform convex mapping (default is concave).");
        Console.WriteLine("{0,-23} {1}", "  -bx:{int} | -bx={int}", "Left & right border size, in pixels (default is 0).");
        Console.WriteLine("{0,-23} {1}", "  -by:{int} | -by={int}", "Top & bottom border size, in pixels (default is 0).");
        Console.WriteLine("{0,-23} {1}", "  -b:{int}  | -b={int}", "Border size, in pixels (default is 0).");
    }
}
