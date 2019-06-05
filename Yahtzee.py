import sys
ls = []
ls.append(int(sys.argv[1]))
ls.append(int(sys.argv[2]))
ls.append(int(sys.argv[3]))
ls.sort()
summ = ls[0]+ls[1]+ls[2]


####all equal
if ls[0] == ls[2]:
    print("No change is needed")

def exp(ls):
    ## change 1
    exp = ls[0] + ls[1] + ls[2]
    print(exp)
    back = 0
    if ls[0] != ls[1]:
        if ls[1] != ls[2]:
            exp1 = (1/6 + 2/6 + 3/6 + 4/6 + 5/6 + 6/6) + ls[1] + ls[2]
        else:
            exp1 = (1/6 + 2/6 + 3/6 + 4/6 + 5/6 + 6/6) + 2*ls[1] + 25/6 - ls[1]/2
    else:
        exp1back = (1/6 + 2/6 + 3/6 + 4/6 + 5/6 + 6/6) + 2*ls[1] + 25/6 - ls[1]/2
        exp1 = (1/6 + 2/6 + 3/6 + 4/6 + 5/6 + 6/6) + ls[1] + ls[2]
        if exp1 < exp1back:
            back = 1
            exp1 = exp1back
    print(exp1)
    ## change 2
    exp2 = (1/6 + 2/6 + 3/6 + 4/6 + 5/6 + 6/6)*2+ ls[2] + 25/36 - ls[2]/12
    print(exp2)
    ## change 3
    exp3 = (1/6 + 2/6 + 3/6 + 4/6 + 5/6 + 6/6)*3 + (25/36) - (21/72)
    print(exp3)
    if exp > exp1 and exp > exp2 and exp > exp3:
        print("No change is needed")
    if exp1 > exp2 and exp1 > exp3:
        if back == 1:
            print("change value: "+str(ls[2]))
        else:
            print("change value: "+str(ls[0]))
    if exp2 > exp1 and exp2 > exp3:
        print("change value: "+str(ls[0])+" "+str(ls[1]))
    if exp3 > exp1 and exp3 > exp2:
        print("change value: "+str(ls[0])+" "+str(ls[1])+" "+str(ls[2]))

exp(ls)
