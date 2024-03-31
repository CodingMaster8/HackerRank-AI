import math


def product_xy(data1, data2):
    sum_xy = 0
    for i in range(len(data1)):
        sum_xy += data1[i] * data2[i]
    return sum_xy



def sum_data(data):
    sum_data = 0
    for i in range(len(data)):
        sum_data += data[i]
    return sum_data


def sum_data_squared(data):
    sum_data = 0
    for i in range(len(data)):
        sum_data += (data[i] ** 2)
    return sum_data


physics = [15.0, 12.0, 8.0, 8.0, 7.0, 7.0, 7.0, 6.0, 5.0, 3.0]
history = [10.0, 25.0, 17.0, 11.0, 13.0, 17.0, 20.0, 13.0, 9.0, 15.0]

sum_xy = product_xy(physics, history)
sum_x = sum_data(physics)
sum_y = sum_data(history)
sum_squared_x = sum_data_squared(physics)
sum_squared_y = sum_data_squared(history)

divider = sum_xy - (sum_x * sum_y)
dividend = math.sqrt((sum_squared_x - (sum_x ** 2) * (sum_squared_y - (sum_y ** 2))))

r = divider / dividend

print(round(r,3))