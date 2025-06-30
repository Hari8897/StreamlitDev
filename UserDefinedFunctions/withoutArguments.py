# Functon without Arguments


def hello():
    print("Welcom to Python Programing")


def area_square():
    side = eval(input("Enter the side of Squre: "))
    print("area of square: ",side**2)

def area_circle():
    rad=eval(input("Enter radius of circle: "))
    print("area of cicle: ",3.14*rad**2)
def area_rectangle():
    length,breadth =eval(input("Input the length and breadth: "))
    print("The are of rectingle: ",length*breadth)
def area_traingle():
    a,b,c =eval(input("Input the three sides of traingle: "))
    s = (a+b+c)/2
    print("The area of the triangle is: ",(((s-a)+(s-b)+(s-c))**0.5))
    
choice = int(input("Choice: 1-square, 2-circle, 3-regtangle, 4-traingle:"))
if choice ==1:
    area_square()
elif choice ==2:
    area_circle()
elif choice == 3:
    area_rectangle()
elif choice ==4:
    area_traingle()
else:
    print("Not a valid choice")






    

