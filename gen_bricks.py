import tkinter as tk
from random import choice

# 游戏窗口大小
WIDTH = 800
HEIGHT = 600

# 球和挡板的速度
BALL_SPEED_X = 4
BALL_SPEED_Y = -4
PADDLE_SPEED = 30

class BrickBreaker:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='black')
        self.canvas.pack()

        # 创建球
        self.ball = self.canvas.create_oval(WIDTH // 2 - 15, HEIGHT // 2 - 15,
                                            WIDTH // 2 + 15, HEIGHT // 2 + 15,
                                            fill='white')

        # 创建挡板
        self.paddle = self.canvas.create_rectangle(WIDTH // 2 - 60, HEIGHT - 30,
                                                   WIDTH // 2 + 60, HEIGHT - 10,
                                                   fill='blue')

        # 创建砖块
        self.bricks = []
        brick_colors = ['red', 'orange', 'yellow', 'green']
        for row in range(5):
            for col in range(10):
                x1 = col * (WIDTH // 10)
                y1 = row * 25 + 50
                x2 = x1 + (WIDTH // 10)
                y2 = y1 + 25
                color = choice(brick_colors)
                brick = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
                self.bricks.append(brick)

        # 设置初始速度
        self.speed_x = BALL_SPEED_X
        self.speed_y = BALL_SPEED_Y

        # 绑定键盘事件
        self.root.bind('<Left>', self.move_paddle_left)
        self.root.bind('<Right>', self.move_paddle_right)

        # 开始游戏循环
        self.game_loop()

    def move_paddle_left(self, event):
        pos = self.canvas.coords(self.paddle)
        if pos[0] > 0:
            self.canvas.move(self.paddle, -PADDLE_SPEED, 0)

    def move_paddle_right(self, event):
        pos = self.canvas.coords(self.paddle)
        if pos[2] < WIDTH:
            self.canvas.move(self.paddle, PADDLE_SPEED, 0)

    def game_loop(self):
        self.move_ball()
        self.check_collision()
        self.root.after(20, self.game_loop)

    def move_ball(self):
        self.canvas.move(self.ball, self.speed_x, self.speed_y)

    def check_collision(self):
        ball_pos = self.canvas.coords(self.ball)
        paddle_pos = self.canvas.coords(self.paddle)

        # 检查与上边界的碰撞
        if ball_pos[1] <= 0:
            self.speed_y = -self.speed_y

        # 检查与下边界的碰撞（游戏结束）
        if ball_pos[3] >= HEIGHT:
            self.game_over()

        # 检查与左边界的碰撞
        if ball_pos[0] <= 0 or ball_pos[2] >= WIDTH:
            self.speed_x = -self.speed_x

        # 检查与挡板的碰撞
        if ball_pos[3] >= paddle_pos[1] and ball_pos[0] <= paddle_pos[2] and ball_pos[2] >= paddle_pos[0]:
            self.speed_y = -self.speed_y

        # 检查与砖块的碰撞
        for brick in self.bricks:
            brick_pos = self.canvas.coords(brick)
            if ball_pos[2] >= brick_pos[0] and ball_pos[0] <= brick_pos[2] \
                    and ball_pos[3] >= brick_pos[1] and ball_pos[1] <= brick_pos[3]:
                self.canvas.delete(brick)
                self.bricks.remove(brick)
                self.speed_y = -self.speed_y
                break

    def game_over(self):
        self.canvas.create_text(WIDTH // 2, HEIGHT // 2, text='Game Over',
                                font=('Helvetica', 30), fill='white')
        self.root.unbind('<Left>')
        self.root.unbind('<Right>')

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Brick Breaker')
    game = BrickBreaker(root)
    root.mainloop()