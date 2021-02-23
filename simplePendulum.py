# simplePendulum.py
# Author: Dinusha Senarathna
# Unp.sing Euler Method
import numpy as np
import pygame
import matplotlib.pyplot as plt

pygame.init()
win = pygame.display.set_mode((1000,500))
white = [255, 255, 255]
win.fill(white)
pygame.display.set_caption('Simple Pendulum')

# Problem parameters
g=9.81 # gravitational acceleration (N/m)
l=0.1 # string length (m)
alpha = 3
theta_init = np.pi/3 # initial angular displacement (rad)
theta_dot_init = 0 # initial angular velosity (rad/s)

# Set time step stuff
end_t=20 # simulation time (s)
delta_t=0.0001 # simulation time step
frameRate = int(1/delta_t)
iterations=int(end_t/delta_t)
t=np.linspace(0,1,iterations)
cnt = 0

# Pre-allocate variables for speed
theta=np.zeros(iterations)
theta[0]=theta_init
x=np.zeros(iterations)
x[0]=500+l*np.sin(theta_init)*3000
y=np.zeros(iterations)
y[0]=l*np.cos(theta_init)*3000
theta_dot=np.zeros(iterations)
theta_dot[0]=theta_dot_init
theta_dbl_dot=np.zeros(iterations)
theta_dbl_dot[0]=(-g/l)*np.sin(theta_init)-alpha*l*theta_dot_init

# Set up video
class blob(object):
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.vel_x = 0
        self.vel_y = 0

    def draw(self, win):
        pygame.draw.circle(win, (0,0,255), (int(self.x), int(self.y)), self.r, 5)
        pygame.draw.line(win, (255,0,0), (500,0), (int(self.x), int(self.y)), 2)

# Begin iterative process of solving the ODE's with Euler's Method
for n in range(1,iterations):    
  theta[n]=theta[n-1]+theta_dot[n-1]*delta_t
  x[n]=500+l*np.sin(theta[n])*3000
  y[n]=l*np.cos(theta[n])*3000
  theta_dot[n]=theta_dot[n-1]+theta_dbl_dot[n-1]*delta_t
  theta_dbl_dot[n]=(-g/l)*np.sin(theta[n])-alpha*l*theta_dot[n]
  
# Plot everything for the video
f, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(t, x)
ax1.set_title('x-t')
ax2.plot(t, y)
ax2.set_title('y-t')
ax3.plot(t, theta)
ax3.set_title('theta-t')

#ax1.legend()
#ax2.legend()
#ax3.legend()
plt.show()

rock = blob(x[0], y[0], 10)
def redrawGameWindow():
    rock.draw(win)
    pygame.display.update()

clock = pygame.time.Clock()
run = True
while run:
    clock.tick(100)    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    if iterations-1 < cnt:
        cnt = cnt-1
        run = False
    rock.x = x[cnt]
    rock.y = y[cnt]
    cnt += 100
    win.fill(white)
    redrawGameWindow()
pygame.quit()