import math
import pandas
import numpy as np
import random

ST = [9, 10.5, 12, 13.5, 15]  # 当地时间
D = [305, 336, 0, 31, 61, 92, 122, 153, 184, 214, 244, 275]  # 距离3月21日的天数
phi = 39.4 * math.pi / 180  # phi是纬度
H = 3000  # 海拔
ita_cos = 0  # 余弦效率
ita_shadow = 0  # 阴影效率
ita_truncation = 0  # 截断效率
ita_atmosphere = 0  # 大气透射率
total_cos = 0.0  # 总余弦效率
total_shadow = 0.0  # 总阴影效率
total_truncation = 0.0  # 总截断效率
total_atmosphere = 0.0  # 总大气透射率
reflectors = []

class Reflector:
    x = 0  # 以东方向为x轴正半轴
    y = 0  # 以北方向为y轴正半轴
    R = 0  # 反射镜中心到塔中心的距离
    h = 0  # 反射镜高度
    a = 0  # 反射镜高
    b = 0  # 反射镜宽
    beta = 0  # 反射镜高度到接收器仰角
    alpha = 0  # 反射镜俯仰角
    gamma = 0  # 反射镜方位角
    v_n = np.array([0, 0, 0])  # 反射镜法向量
    v_refSunlight = np.array([0, 0, 0])  # 反射光线的方向向量
    T = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # 镜面坐标系转地面坐标系矩阵
    O = np.array([x, y, h])  # 反射镜中心坐标

    def __init__(self, x, y, h, a, b, sunlight1):
        self.x = x
        self.y = y
        self.R = math.sqrt(x ** 2 + y ** 2)
        self.h = h
        self.a = a
        self.b = b
        self.beta = math.atan((80 - self.h) / self.R)
        self.alpha = (sunlight1.alphaS - self.beta) / 2
        self.v_refSunlight = np.array([-self.x / (self.x ** 2 + self.y ** 2 + (80 - self.h) ** 2) ** 0.5,
                                       -self.y / (self.x ** 2 + self.y ** 2 + (80 - self.h) ** 2) ** 0.5,
                                       (80 - self.h) / (self.x ** 2 + self.y ** 2 + (80 - self.h) ** 2) ** 0.5])
        self.O = np.array([self.x, self.y, self.h])
        # 计算两个向量的模
        l_refSunlight = np.sqrt(self.v_refSunlight.dot(self.v_refSunlight))
        l_sunlight = np.sqrt(sunlight1.v_sunlight.dot(sunlight1.v_sunlight))

        # 计算两个向量的点积
        dot = self.v_refSunlight.dot(sunlight1.v_sunlight)

        # 计算两个向量夹角的余弦值
        cos_ = dot / (l_refSunlight * l_sunlight)

        if cos_ < 0:
            self.v_n = (-sunlight1.v_sunlight) + self.v_refSunlight
        else:
            self.v_n = sunlight1.v_sunlight + self.v_refSunlight

        self.gamma = math.atan(self.v_n[0] / self.v_n[1])
        self.T = np.array([[math.cos(self.gamma), math.cos(self.alpha) * math.sin(self.gamma),
                            -math.cos(self.alpha) * math.sin(self.gamma)],
                           [-math.sin(self.gamma), math.cos(self.alpha) * math.cos(self.gamma),
                            -math.cos(self.alpha) * math.cos(self.gamma)],
                           [0, math.sin(self.alpha), math.cos(self.alpha)]
                           ])


class SunLight:
    alphaS = 0  # 太阳高度角
    gammaS = 0  # 太阳方位角
    omega = 0  # 太阳时角
    delt = 0  # 太阳赤纬角
    v_sunlight = np.array([0, 0, 0])  # 太阳光线的方向向量

    def __init__(self, ST, D, phi):
        self.omega = math.pi * (ST - 12) / 12
        self.delt = math.asin(math.sin(2 * math.pi * D / 365) * math.sin(23.45 * math.pi / 180))
        self.alphaS = math.asin(
            math.sin(phi) * math.sin(self.delt) + math.cos(phi) * math.cos(self.delt) * math.cos(self.omega))
        self.gammaS = math.acos(
            (math.sin(self.delt) - math.sin(self.alphaS) * math.sin(phi)) / math.cos(self.alphaS) * math.cos(phi))
        self.v_sunlight = np.array(
            [math.cos(self.alphaS) * math.sin(self.gammaS), math.cos(self.alphaS) * math.cos(self.gammaS),
             math.sin(self.alphaS)])


def createTower():
    x = random.uniform(-350, 350)
    y = random.uniform(-350, 350)
    while (x**2 + y**2) > 122500:
        x = random.uniform(-350, 350)
        y = random.uniform(-350, 350)

    return x, y


x_tower, y_tower = createTower()


def createReflectors(num, x_t, y_t, sunlight):
    reflectors = []

    # 生成镜面高度和镜面宽度
    length = random.uniform(2, 8)
    width = random.uniform(2, 8)

    # 确保镜面宽度大于等于镜面高度
    while width < length:
        length = random.uniform(2, 8)
        width = random.uniform(2, 8)

    height = random.uniform(max(2.0, length / 2), 6)

    coordinates = []

    # 生成第一个坐标
    x = random.uniform(-350, 350)
    y = random.uniform(-350, 350)
    while ((x - x_t)**2 + (y - y_t)**2) < 10000 or (x**2 + y**2) > 122500:
        x = random.uniform(-350, 350)
        y = random.uniform(-350, 350)
    coordinates.append([x, y, height, length, width])

    # 生成剩余的坐标
    for _ in range(num - 1):
        while True:
            # 随机生成新的坐标
            x = random.uniform(-350, 350)
            y = random.uniform(-350, 350)

            # 检查与之前的坐标的距离是否大于等于镜面宽度加5并且是否在圆环内
            valid = True
            for coord in coordinates:
                dist = math.sqrt((x - coord[0]) ** 2 + (y - coord[1]) ** 2)
                if dist < (length + 5) or ((x - x_t)**2 + (y - y_t)**2) < 10000 or (x**2 + y**2) > 122500:
                    valid = False
                    break

            # 如果距离满足条件，则添加坐标并退出循环
            if valid:
                coordinates.append([x, y, height, length, width])
                break

    # 创建反射镜实例添加进反射镜组
    for coord in coordinates:
        reflector = Reflector(coord[0], coord[1], height, length, width, sunlight)
        reflectors.append(reflector)
    return reflectors


def calCos(reflectors, sunlight):
    S = []
    for reflector in reflectors:
        cos2theta = (-reflector.x * math.cos(sunlight.alphaS) * math.sin(sunlight.gammaS) - reflector.y * math.cos(
            sunlight.alphaS) * math.cos(sunlight.gammaS) + reflector.h * math.sin(sunlight.alphaS)) / math.sqrt(
            reflector.R ** 2 + reflector.h ** 2)
        S.append(math.sqrt((1 + cos2theta) / 2))
    return S


def calAtmosphere(reflectors):
    S = []
    for reflector in reflectors:
        d_HR = math.sqrt(reflector.R**2 + (80 - reflector.h)**2)
        ita_at = 0.99321 - 0.0001176*d_HR + 1.97e-8 * d_HR**2
        S.append(ita_at)
    return S


def calShadowArea(reflectors, sunlight):
    S = []  # 阴影面积
    ita = []
    for reflector1 in reflectors:
        for reflector2 in reflectors:
            if (reflector1.x == reflector2.x and reflector1.y == reflector2.y) or (
                    (reflector1.x - reflector2.x) ** 2 + (
                    reflector1.y - reflector2.y) ** 2 > 400) or reflector1.R > reflector2.R:
                continue
            else:
                # 取前反射镜的左右顶点
                H1_reflector1 = np.array([-3, 3, 0])
                H2_reflector1 = np.array([3, 3, 0])

                # 将两点转换到地面坐标系下
                H1_ground = reflector1.T.dot(H1_reflector1) + reflector1.O
                H2_ground = reflector1.T.dot(H2_reflector1) + reflector1.O

                # 将两点转换到后反射镜坐标系下
                H1_reflector2 = reflector2.T.T.dot(H1_ground - reflector2.O)
                H2_reflector2 = reflector2.T.T.dot(H2_ground - reflector2.O)

                # 将光线在地面坐标系下的方向向量转换到后反射镜坐标系下
                v_sunlight_reflector2 = reflector2.T.T.dot(sunlight.v_sunlight)

                # 获得光线与后反射镜的交点
                a = v_sunlight_reflector2[0]
                b = v_sunlight_reflector2[1]
                c = v_sunlight_reflector2[2]
                x1 = H1_reflector2[0]
                y1 = H1_reflector2[1]
                z1 = H1_reflector2[2]
                x2 = H2_reflector2[0]
                y2 = H2_reflector2[1]
                z2 = H2_reflector2[2]

                H1 = np.array([(c * x1 - a * z1) / c, (c * y1 - b * z1) / c, 0])
                H2 = np.array([(c * x2 - a * z2) / c, (c * y2 - b * z2) / c, 0])

                # 判断交点是否在后反射镜内
                if (-3 < H1[0] < 3) and (-3 < H1[1] < 3):
                    s = (3 - H1[0]) * (3 - H1[1])
                    S.append(s/reflector2.a*reflector2.b)
                elif (-3 < H2[0] < 3) and (-3 < H2[1] < 3):
                    s = -(3 - H2[0]) * (-3 - H2[1])
                    S.append(s/reflector2.a*reflector2.b)
    return S


def calTruncationArea(reflectors, sunlight):
    S = []  # 截断面积
    for reflector in reflectors:
        a = -reflector.x
        b = -reflector.y
        c = -reflector.h + 80

        H1 = reflector.T.dot(np.array([-reflector.b/2, -reflector.a/2, 0])) + reflector.O
        H2 = reflector.T.dot(np.array([-reflector.b/2, reflector.a/2, 0])) + reflector.O
        H3 = reflector.T.dot(np.array([reflector.b/2, -reflector.a/2, 0])) + reflector.O
        H4 = reflector.T.dot(np.array([reflector.b/2, reflector.a/2, 0])) + reflector.O

        k = a * a + b * b + c * c

        t1 = (-a * H1[0] - b * H1[1] - c * H1[2]+a*x_tower+b*y_tower) / k
        t2 = (-a * H2[0] - b * H2[1] - c * H2[2]+a*x_tower+b*y_tower) / k
        t3 = (-a * H3[0] - b * H3[1] - c * H3[2]+a*x_tower+b*y_tower) / k
        t4 = (-a * H4[0] - b * H4[1] - c * H4[2]+a*x_tower+b*y_tower) / k

        A=np.array([H1[0] + t1 * a, H1[1] + t1 * b, H1[2] + t1 * c])
        B=np.array([H2[0] + t2 * a, H2[1] + t2 * b, H2[2] + t2 * c])
        C=np.array([H3[0] + t3 * a, H3[1] + t3 * b, H3[2] + t3 * c])
        D=np.array([H4[0] + t4 * a, H4[1] + t4 * b, H4[2] + t4 * c])

        AB=B-A
        AC=C-A
        BD=D-B

        cross_product=np.cross(AB,AC)
        width=np.linalg.norm(cross_product)/np.linalg.norm(BD)
        distance = np.linalg.norm(cross_product) / np.linalg.norm(AB)
        Distance=distance*math.sqrt(k)/c

        angle_in_radians = math.radians(0.53/2)
        tan_value = math.tan(angle_in_radians)
        light = tan_value*math.sqrt(76*76+(reflector.x-x_tower)**2+(reflector.y-y_tower)**2)
        S.append(7 * 8 / ((width+2*light)*(Distance+2*light)))
    return S


def totalCal(reflectors, sunlight):
    a = 0.4237 - 0.00821 * (6 - H)**2
    b = 0.5055 + 0.00595 * (6.5 - H)**2
    c = 0.2711 + 0.01858 * (2.5 - H)**2
    g0 = 1.366
    DNI = g0 *(a + b * math.exp(-c / math.cos(sunlight.alphaS)))
    ita_c = calCos(reflectors, sunlight)
    ita_s = calShadowArea(reflectors, sunlight)
    ita_t = calTruncationArea(reflectors, sunlight)
    ita_a = calAtmosphere(reflectors)

    E_field = 0
    ita = []
    for i in range(len(reflectors)):
        ita.append(0.92 * ita_c[i] * ita_s[i] * ita_t[i] * ita_a[i])
        E_field += DNI * reflectors[i].a * reflectors[i].b * ita[i]
    return E_field
