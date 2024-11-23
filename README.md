# wsim
A wasm uav simulator
![result (1)](https://github.com/user-attachments/assets/b510d469-31b7-4f84-bbca-a2ed5a686356)

sudo apt update
sudo apt install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev


drone - simulation
rasterizer enables realtime rendering
raytracer makes lighting natural
real time raytracer in under 1k LOC would be a great base

then a proper 3D model would be about time

either we screw visual realism for now 
or this turns more and more into simulator.ratisbonrobotics.com

again parts are: renderer, dynamics, collision, drone model, environment model
we kind of have everything somewhat - it just needs to be pieced together

for now we should not have any hardware requirements - rendering output is simply an array of pixels
