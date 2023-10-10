import math
import pandas
import numpy as np

coordinates = pandas.read_excel('附件.xlsx').values
ST = [9, 10.5, 12, 13.5, 15] # 当地时间
D = [305, 336, 0, 31, 61, 92, 122, 153, 184, 214, 244, 275] # 距离3月21日的天数
phi = 39.4 * math.pi/180    # phi是纬度
ita_cos = 0  # 余弦效率
ita_shadow = 0  # 阴影效率
ita_truncation = 0  # 截断效率
ita_atmosphere = 0  # 大气透射率
total_cos = 0.0  # 总余弦效率
total_shadow = 0.0  # 总阴影效率
total_truncation = 0.0  # 总截断效率
total_atmosphere = 0.0  # 总大气透射率
a=0.4237-0.00821*(6-3)**2
b=0.5055+0.00595*(6.5-3)**2
c=0.2711+0.01858*(2.5-3)**2
E=0



class Reflector:
    x = 0 # 以东方向为x轴正半轴
    y = 0 # 以北方向为y轴正半轴
    R = 0 # 反射镜中心到塔中心的距离
    h = 0 # 反射镜高度
    beta = 0  # 反射镜高度到接收器仰角
    alpha = 0  # 反射镜俯仰角
    gamma = 0  # 反射镜方位角
    v_n = np.array([0, 0, 0])  # 反射镜法向量
    v_refSunlight = np.array([0, 0, 0])  # 反射光线的方向向量
    T = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # 镜面坐标系转地面坐标系矩阵
    O = np.array([x, y, h])  # 反射镜中心坐标

    def __init__(self, x, y, h, sunlight1):
        self.x = x
        self.y = y
        self.R = math.sqrt(x**2 + y**2)
        self.h = h
        self.beta = math.atan((80-self.h)/self.R)
        self.alpha = (sunlight1.alphaS - self.beta)/2
        self.v_refSunlight = np.array([-self.x/(self.x**2 + self.y**2 + (80-self.h)**2)**0.5, -self.y/(self.x**2 + self.y**2 + (80-self.h)**2)**0.5, (80-self.h)/(self.x**2 + self.y**2 + (80-self.h)**2)**0.5])
        self.O = np.array([self.x, self.y, self.h])
        # 计算两个向量的模
        l_refSunlight = np.sqrt(self.v_refSunlight.dot(self.v_refSunlight))
        l_sunlight = np.sqrt(sunlight1.v_sunlight.dot(sunlight1.v_sunlight))

        # 计算两个向量的点积
        dot = self.v_refSunlight.dot(sunlight1.v_sunlight)

        # 计算两个向量夹角的余弦值
        cos_ = dot / (l_refSunlight * l_sunlight)

        if cos_ < 0 :
            self.v_n = (-sunlight1.v_sunlight) + self.v_refSunlight
        else:
            self.v_n = sunlight1.v_sunlight + self.v_refSunlight

        self.gamma = math.atan(self.v_n[0]/self.v_n[1])
        self.T = np.array([[math.cos(self.gamma), math.cos(self.alpha)*math.sin(self.gamma), -math.cos(self.alpha)*math.sin(self.gamma)],
                             [-math.sin(self.gamma), math.cos(self.alpha)*math.cos(self.gamma), -math.cos(self.alpha)*math.cos(self.gamma)],
                             [0, math.sin(self.alpha), math.cos(self.alpha)]
                             ])

class SunLight:
    alphaS = 0  # 太阳高度角
    gammaS = 0  # 太阳方位角
    omega = 0  # 太阳时角
    delt = 0  # 太阳赤纬角
    v_sunlight = np.array([0, 0, 0])  # 太阳光线的方向向量

    def __init__(self, ST, D, phi):
        self.omega = math.pi*(ST - 12)/12
        self.delt = math.asin(math.sin(2*math.pi*D/365)*math.sin(23.45*math.pi/180))
        self.alphaS = math.asin(math.sin(phi)*math.sin(self.delt) + math.cos(phi)*math.cos(self.delt)*math.cos(self.omega))
        self.gammaS = math.acos((math.sin(self.delt) - math.sin(self.alphaS)*math.sin(phi))/math.cos(self.alphaS)*math.cos(phi))
        self.v_sunlight = np.array([math.cos(self.alphaS)*math.sin(self.gammaS), math.cos(self.alphaS)*math.cos(self.gammaS), math.sin(self.alphaS)])


def ita_Cos(reflector, sunlight):
    cos2theta = (-reflector.x * math.cos(sunlight.alphaS) * math.sin(sunlight.gammaS) - reflector.y * math.cos(
        sunlight.alphaS) * math.cos(sunlight.gammaS) + reflector.h * math.sin(sunlight.alphaS)) / math.sqrt(
        reflector.R ** 2 + reflector.h ** 2)
    return math.sqrt((1 + cos2theta) / 2)


def calAtmosphere(reflectors):
    total_at = 0.0
    for reflector in reflectors:
        d_HR = math.sqrt(reflector.R**2 + (80 - reflector.h)**2)
        ita_at = 0.99321 - 0.0001176*d_HR + 1.97e-8 * d_HR**2
        total_at += ita_at
    return total_at/ len(reflectors)


def calCos(reflectors, sunlight):
    S = 0
    for reflector in reflectors:
        S += ita_Cos(reflector, sunlight)
    return S/len(reflectors)


def calShadowArea(reflectors, sunlight):
    S = []  # 阴影面积
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
                    S.append(s)
                elif (-3 < H2[0] < 3) and (-3 < H2[1] < 3):
                    s = -(3 - H2[0]) * (-3 - H2[1])
                    S.append(s)
    return S


def calTruncationArea(reflectors, sunlight):
    S = []  # 截断面积
    for reflector in reflectors:
        a = -reflector.x
        b = -reflector.y
        c = -reflector.h + 80

        H1 = reflector.T.dot(np.array([-3, -3, 0])) + reflector.O
        H2 = reflector.T.dot(np.array([-3, 3, 0])) + reflector.O
        H3 = reflector.T.dot(np.array([3, -3, 0])) + reflector.O
        H4 = reflector.T.dot(np.array([3, 3, 0])) + reflector.O

        k = a * a + b * b + c * c

        t1 = (-a * H1[0] - b * H1[1] - c * H1[2]) / k
        t2 = (-a * H2[0] - b * H2[1] - c * H2[2]) / k
        t3 = (-a * H3[0] - b * H3[1] - c * H3[2]) / k
        t4 = (-a * H4[0] - b * H4[1] - c * H4[2]) / k

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
        light=tan_value*math.sqrt(76*76+reflector.x*reflector.x+reflector.y*reflector.y)
        S.append((width+2*light)*(Distance+2*light))
    return reflectors.__len__() * 8 * 7 / (sum(S))


def calDNIShadow(sunlight):
    return 1.366*(a+b*math.exp(-c/sunlight.alphaS))


for i in range(len(D)):
    energy = 0
    for j in range(len(ST)):
        d = D[i]
        st = ST[j]
        sunlight = SunLight(st, d, phi)
        reflectors = []
        for coordinate in coordinates:
            x = coordinate[0]
            y = coordinate[1]
            reflector = Reflector(x, y, 4, sunlight)
            reflectors.append(reflector)
        ita_E = 0
        ita_E = 1745 / 60 * calDNIShadow(sunlight) * 36 * calCos(reflectors, sunlight) * (
                    1 - sum(calShadowArea(reflectors, sunlight)) / (reflectors.__len__() * 6 * 6)) * calTruncationArea(
            reflectors, sunlight) * calAtmosphere(reflectors)
        energy+=ita_E*60
        E += ita_E

        ita_cos = 0
        ita_shadow = 0
        ita_truncation = 0
        ita_atmosphere = 0

        ita_cos += calCos(reflectors, sunlight)
        ita_shadow += (1 - sum(calShadowArea(reflectors, sunlight))/(reflectors.__len__()*6*6))
        ita_truncation += (calTruncationArea(reflectors, sunlight))
        ita_atmosphere += calAtmosphere(reflectors)

    print("{}月21日的单位面积镜面平均输出热功率为{:.4f}".format(i+1, energy/5/36/1745))
    total_cos += ita_cos/12
    total_shadow += ita_shadow/12
    total_truncation += ita_truncation/12
    total_atmosphere += ita_atmosphere/12
    print("{}月21日的光学效率为{:.4f},余弦效率为{:.4f},阴影效率为{:.4f},截断效率为{:.4f},大气透射率为{:.4f}".format(i+1, ita_cos*ita_shadow*ita_truncation*ita_atmosphere*0.92, ita_cos, ita_shadow, ita_truncation, ita_atmosphere))
print("年平均光学效率为{:.4f},年平均余弦效率为{:.4f},年平均阴影效率为{:.4f},年平均截断效率为{:.4f},年平均大气透射率为{:.4f}".format(total_cos*total_truncation*total_shadow*total_atmosphere*0.92, total_cos, total_shadow, total_truncation, total_atmosphere))
print(E)