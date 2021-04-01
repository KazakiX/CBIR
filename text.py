hour = int(input("Enter Hours:"))
date = int(input("Enter Rate:"))
if hour <= 40:
    print("pay:"+ date*hour)
if hour > 40:
    payment = date*40 +(hour-40)*date*1.5
    print("pay:", payment )


try:
    hour = int(input("Enter Hours:"))
    date = int(input("Enter Rate:"))
except ValueError as e:
    print("Error, please enter numeric input")
finally:
    if hour <= 40:
        print("pay:"+ date*hour)
    if hour > 40:
        payment = date*40 +(hour-40)*date*1.5
        print("pay:", payment )


score = float(input("Enter score:"))
if score>=0.9 and score<=1:
    print("A")
elif score>=0.8 and score<0.9:
    print("B")
elif score>=0.7 and score<0.8:
    print("C")
elif score>=0.6 and score<0.7:
    print("D")
elif score>=0 and score<0.6:
    print("F")
else:
    print("Bad score")