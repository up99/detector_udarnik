# detector_udarnik

This repo contains source files of app.

There are 2 options how to launch the progeam:

1. Install all dependecies provided in environment.yml and run [python detector.py] via terminal
2. (For Windows users) Download repo and build exe file via command [pyinstaller detector.py --windowed --add-data "runs\detect\train3\weights\best.pt;runs\detect\train3\weights\best.pt" --clean]