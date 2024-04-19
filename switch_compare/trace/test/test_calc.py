# coding=utf8


def comm_calc(m, pp, v, comp, tot):
    ideal = m * v * comp
    bubble = (pp - 1 ) * comp
    comm_time = tot - ideal - bubble
    print(f"ideal time: {ideal}, buble time: {bubble}, comm time: {comm_time}, ideal ratio: {ideal / tot}")



m = 6
v = 2
comp = 30
tot = 430
pp=3

m = 6
v = 4
tot = 802 

m = 6
v = 8
tot = 1546 

comm_calc(m, pp, v, comp, tot)
