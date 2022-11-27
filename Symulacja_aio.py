import os.path
import scipy.stats
import math
import random
import numpy as np
from PIL import Image
import scipy
import scipy.stats
from numpy import random, sqrt, log, sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib.pyplot import matshow
from pylab import show,hist,subplot,figure

# #1 generowanie zbiorow i podstawowe statystyki, kurtoza i skośność
mean = 0
standard_deviation = 1
# n = 100

def skew_my(data, average, std_dev):
    sum = 0
    n = len(data)
    for i in range(0, n):
        sum += math.pow(((data[i] - average)), 3)
    result = (n * sum / ( (n - 1) * (n - 2) * pow(std_dev, 3)))
    return result

def central_moment_kurt(data, mean_data, std_dev):
    n = len(data)
    sum = 0
    for i in range(0, n):
        sum += (math.pow((data[i] - mean_data)/std_dev, 4))
    kurtos = (((n * (n + 1))/((n - 1) * (n - 2) * (n - 3))) * sum ) - ( (3 * (math.pow((n - 1), 2)))/(( n - 2 ) * ( n - 3) ))
    # curtosis for n > 3!
    return kurtos
print('\t\tDo wczytania danych umieść lokalnie plik " distribution_data.txt " - 2 kolumny wartości (normal dist   uniform dist)\n'
      '\t\t\tOddzielone tabulatorem. Brak pliku spowoduje wygenerowanie nowego zbioru danych\n\t\t#####################\n')

# otwóz plik z danymi rozkładów w formacie => (wart.normal) (warto.uniform)
# lewa kolumna = wartosci normalne
# prawa kolumna = wartosci z rozk. jednostajnego
def openData(distribution_type):
    normal = []
    uniform = []
    with open('datasets_arch/distribution_data.txt', 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        for line in lines:
            normal.append(float(line[0]))
            uniform.append(float(line[1]))

    if distribution_type == 'normal':
        return normal
    if distribution_type == 'uniform':
        return uniform

#left - normal distribution, right - uniform distribution
def saveData(population):
    file_exist = os.path.isfile('datasets_arch/distribution_data.txt')
    if not file_exist:
        n = population
        normal_dataset = np.random.normal(0, 0.1, n)
        uniform_dataset = np.random.uniform(0, 1, n)
        print('Plik z danymi nie istnieje!, Zapisywanie...')
        with open('datasets_arch/distribution_data.txt', 'w+') as f:
            for i in range(0, n):
                f.write(f'{normal_dataset[i]}\t{uniform_dataset[i]}\n')
        print('Zapisano dane.')
        return False
    else:
        if __name__ == '__main__':
            print('Plik ze zbiorami istnieje!')
        return False

check = saveData(100)

normal_dist = openData('normal')
uniform_dist = openData('uniform')

average_normal = np.average(normal_dist)
average_uniform = np.average(uniform_dist)
median_normal = np.median(normal_dist)
median_uniform = np.median(uniform_dist)
deviation_normal = np.std(normal_dist)
deviation_uniform = np.std(uniform_dist)

print(f'Normal distribution stat:\n\n'
      f'\tAverage normal: {average_normal}\n'
      f'\tmedian normal: {median_normal}\n'
      f'\tStandard deviation normal SciPy = {deviation_normal}\n'
      # f'\tKurt normal SciPy = {kurtosis(normal_dist)}\n'
      f'\t\tKurt normal - moja funkcja: {central_moment_kurt(normal_dist, average_normal, deviation_normal)}\n'
      # f'\tSkew normal SciPy = {skew(normal_dist)}\n'
      f'\t\tSkew normal - moja funkcja: {skew_my(normal_dist, average_normal, standard_deviation)}\n'

      f'\n'
      f'Uniform distribution stat:\n\n'
      f'\taverage unifrom: {average_uniform}\n'
      f'\tmedian uniform: {median_uniform}\n'
      f'\tStnd deviation uniform SciPy = {deviation_uniform}\n'
      # f'\tKurt uniform SciPy = {kurtosis(uniform_dist)}\n'
      f'\t\tKurt unifrom - moja funkcja: {central_moment_kurt(uniform_dist, average_uniform, deviation_uniform)}\n'
      # f'\tSkew uniform SciPy = {skew(uniform_dist)}\n'
      f'\t\tSkew uniform - moja funkcja: {skew_my(uniform_dist, average_uniform, standard_deviation)}\n'
)
print('###############################')

# #2 testy

# Chi - suare
# estymator dla normalnego = średnia
# estymatorem dla jednostajnego = podział na grupy s chi kwadrat
# poniżej wartość oczekiwana to średnia z próby
def chiSquareTest(data):
    # crit_avg = (max(data) + min(data)) / 2
    expected = 0
    n = len(data)
    chi = 0
    # wart. oczekiwana = srednia arytm wartości
    for i in range(0, len(data)):
        expected += data[i]/n
    for i in range(0, len(data)):
        chi += (pow((data[i] - expected), 2) / expected)
    p = scipy.stats.chi2.sf(chi, n - 1)
    chiSquareTest.p = p
    return chi, p
chiSquareTest.p = 0

scipy_res = scipy.stats.chisquare(uniform_dist)
print('Test chi: ',chiSquareTest(uniform_dist), f'\nSciPy: {scipy_res}\n')
if chiSquareTest.p < 0.05:
    print('\tDane nie pochodzą z rozkładu jednostajnego\n')
else:
    print('\tDane pochodzą z rozkładu jednostajnego\n')
# jeżeli p_value < 0.05 odzucamy hipotezę H0,
# w przeciwnym wypadku mamy pewnw dowody
# do przyjęcia danej hipotezy

def shapiroWilk(data):
    shapiro_tab = [0.3158, 0.2089, 0.1892, 0.1752, 0.164, 0.1547, 0.1466, 0.1394, 0.1329, 0.127, 0.1215, 0.1163, 0.1115, 0.1069, 0.1026, 0.0984, 0.0944, 0.0906, 0.0869, 0.0834, 0.0834, 0.0765, 0.0733, 0.0701, 0.067, 0.0639, 0.0609, 0.058, 0.0551, 0.0523, 0.0495, 0.0467, 0.044, 0.0413, 0.0387, 0.0361, 0.0335, 0.0309, 0.0284, 0.0258, 0.0233, 0.0208, 0.0183, 0.0159, 0.0134, 0.011, 0.0083, 0.0061, 0.0037, 0.0012]
    sorted_tab = sorted(data)
    m = np.average(data)
    sum_of_y = 0
    b = 0
    for j in range(len(data)):
        sum_of_y += (data[j] - m)**2
    for i in range(1, (len(data)//2) + 1):
        b += shapiro_tab[i-1] * (sorted_tab[len(data) - i] - sorted_tab[i-1])
    b = b**2
    w = b / sum_of_y
    return w

print('Test Shappiro-Wilk: ', shapiroWilk(normal_dist))
print('SciPy', scipy.stats.shapiro(normal_dist))

if scipy.stats.shapiro(normal_dist).pvalue < 0.05:
    print('\tDane nie pochodzą z rozkładu normalnego\n')
else:
    print('\tDane pochodzą z rozkładu normalnego\n')

# WYKRESY
def showPlots():
    figure, axis = plt.subplots(2, 2)
    axis[0, 0].plot(uniform_dist)
    axis[0, 0].set_title('Elementy rozkładu jednostajnego')
    axis[1, 0].set_title('Hist. dla rozk. jednostajnego')
    axis[0, 1].plot(normal_dist)
    axis[0, 1].set_title('Elementy rozkładu normalnego')
    axis[1, 1].set_title('Hist. dla rozk. normalnego')
    axis[1, 0].hist(uniform_dist)
    axis[1, 1].hist(normal_dist)
    plt.show()

showPlots()

# 3 generatory
uniform_origin = openData('uniform')

def rng(m=2**32, a=1103515245, c=12345):
    rng.current = (a*rng.current + c) % m
    return rng.current/m

rng.current = 1 #SEED static variable
uni1 = [rng() for i in range(100)]
uni2 = [rng() for i in range(100)]

test = scipy.stats.wilcoxon(uniform_origin, uni2)
# print(test)
print('Checking Uniform distribution from data sets:')
if test.pvalue > 0.05:
    print(f'UNIFORM - Data from the same distribution p={test.pvalue} > 0.05\n')
else:
    print(f'Different distributions p={test.pvalue} < 0.05\n')

# print(uni1)
# print(uni2)
# print(uniform_origin)

# U1 = np.random.uniform(size = 100) #TEST
# Unormal = np.random.normal(size=100) #TEST

# transformation function
def gaussian(u1,u2):
  z1 = sqrt(-2*log(u1))*cos(2*pi*u2)
  z2 = sqrt(-2*log(u1))*sin(2*pi*u2)
  return z1,z2

u1 = np.array(uni1)

u2 = np.array(uniform_origin)
# print(u1, u2)
# run the transformation
z1,z2 = gaussian(u1,u2)

figure()
subplot(221) # the first row of graphs
hist(u1)     # histograms of u1 and u2
subplot(222)
hist(u2)
subplot(223) # the second contains
hist(z1)     # histograms of z1 and z2
subplot(224)
hist(z2)
show()

normal_test_z1 = scipy.stats.ttest_ind(z1, openData('normal'))
normal_test_z2 = scipy.stats.ttest_ind(z2, openData('normal'))
print('Checking Normal distribution data sets: (Transform)')
if normal_test_z1.pvalue > 0.05:
    print(f'Z1 NORMAL - Data from the same distribution p={normal_test_z1.pvalue} > 0.05')
else:
    print(f'Z1 Data from different distribution p ={normal_test_z1.pvalue} < 0.05')

if normal_test_z2.pvalue > 0.05:
    print(f'Z2 NORMAL - Data from the same distribution p={normal_test_z2.pvalue} > 0.05\n')
else:
    print(f'Z2 Data from different distributions p ={normal_test_z2.pvalue} < 0.05\n')

# print(normal_test_z1)
# print(normal_test_z2)

# 4 - Monte Carlo
print('\n\t##### MONTE CARLO ###')

def f(x):
    return 1/math.sqrt(math.pow(x, 5)+8)
    # return 1/pow(x, 2)
    # return (pow(x, 2) + 2*x)
# zakres
a = 1
b = 6
# dokładność
n = 1000

xs = np.linspace(a, b, n)
print(len(xs))
ys = []

# zakres wartości
y_min = 99999
y_max = 0

for i in xs:
    y_value = f(i)
    ys.append(y_value)
    if y_value > y_max:
        y_max = y_value
    if y_value < y_min:
        y_min = y_value

y_min = math.floor(y_min)

n_grid = 100000
x_grid = np.linspace(a, b, n_grid)
y_grid = np.linspace(y_min, y_max, n_grid)
x_point = []

for i in range(0, n_grid):
    x_point.append(random.uniform(a, b))

y_point = [random.uniform(y_min, y_max) for i in range(n_grid)]
xy_hit = zip(x_point, y_point)

# pole obszaru strzelania
dy = abs(y_min - y_max)
dx = abs(xs[0] - xs[-1])
# print(xs)
print(dx, dy)

sum_of_f_value = 0
for i in xy_hit:
    x_hit = i[0]
    y_hit = i[1]
    y_f = f(x_hit)
    sum_of_f_value += y_f
    # if y_hit >= y_f:
    #     sum_of_f_value += y_f
monte_carlo = (b - a) * (sum_of_f_value/n_grid)
print(f'Monte Carlo: {monte_carlo}')

plt.plot(xs, ys, color='red')
plt.scatter(x_point, y_point)
plt.show()

riemann_d = 1000
def riemman(function, a, b, i):
    dx = (b - a) / i
    integrate = 0
    for x in range(i):
        x = x * dx + a
        integrate += dx * eval(function)
    return integrate

print(f'Riemann: {riemman("1/math.sqrt(pow(x, 5) + 8)", 1, 6, riemann_d)}')

# pole figury asymetrycznej
# obraz z białym tłem, figura z czarnym wypełnieniem
# sprawdzam realną ilość px wypełniającą figurę - dokładna
# siatka pomocnicza to osobny piksel obrazu
# w tym przypadku siatka 477 x 269
im = Image.open('figure_asymetric_jpg.jpg')
pix = im.load()
pix_val = list(im.getdata())
# print(pix_val)
print('\n\t#### Pole FIGURY ###')
print(f'Width: {im.size[0]}\n'
      f'Height: {im.size[1]}\n'
      f'H * W = {im.size[0] * im.size[1]}')
# print(len(pix_val))
inner = 0
# sprawdzamy piksele, które są czarne.

xpx = np.array([i for i in range(1, im.size[0] + 1)])
ypx = np.array([i for i in range(1, im.size[1] + 1)])

px_pos = zip(xpx, ypx)

resolution = 1000000
xpx_rand = np.array([random.uniform(0, im.size[0]) for i in range(resolution)])
ypx_rand = np.array([random.uniform(0, im.size[1]) for i in range(resolution)])
px_rand_zip = zip(xpx_rand, ypx_rand)

plt.scatter(xpx_rand, ypx_rand, color='green')
plt.title('wszystkie strzały w obraz')
plt.show()
# print(xpx)
transformed_img = []
figure_pxs = 0
for pix in pix_val:
    if pix == (0,0,0):
        transformed_img.append(1)
        figure_pxs += 1
    else:
        transformed_img.append(0)
img_matrice = np.reshape(transformed_img, (-1, 477))
print(img_matrice)
matshow(img_matrice)
plt.show()

print(max(ypx_rand), len(ypx_rand))
hitted = 0
hitted_pxs = []
for px in (px_rand_zip):
    # print(px)
    randx = int(px[0])
    randy = int(px[1])
    # print(randx, randy)
    if img_matrice[randy, randx] == 1:
        hitted += 1
        hitted_pxs.append(px)

print(hitted)
hitted_xs = [p[0] for p in hitted_pxs]
hitted_ys = [p[1] for p in hitted_pxs]

plt.scatter(hitted_xs, hitted_ys)
plt.show()
# for p in range(0, len(pix_val)):
result = hitted/resolution * (im.size[0] * im.size[1])
print(f'Real px area: {figure_pxs}')
print(f'Monte Carlo Area: {result} dla n = {resolution}\n')
print(f'W celu większej dokładnośći, zwiększyć resolution')
# print(f'Inner: {inner}')
# print(f'Area: {area}px')