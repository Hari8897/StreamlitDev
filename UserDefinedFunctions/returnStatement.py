# Creating a function without arguments and returning a single value.
from doctest import debug
from multiprocessing import Value


def bill():
    print("Welcome")
    amount=4000
    return amount

# Calling a function.
Value = bill()
print(Value)

# Create a function that returns a Boolean value.
def multiple():
    num1, num2 = eval(input("Input the two numbers: "))
    if num1%num2==0:
        return True
    else:
        return False

# Calling function
# print("The number is multiple: ", multiple())


# Program for using return  i n multiple functions in a single program.
def area():  
    def area_square():
        side = eval(input("Enter the side of Squre: "))
        return side*side

    def area_circle():
        rad=eval(input("Enter radius of circle: "))
        return 3.14*rad**2
    def area_rectangle():
        length,breadth =eval(input("Input the length and breadth: "))
        return length*breadth
    def area_traingle():
        a,b,c =eval(input("Input the three sides of traingle: "))
        s = (a+b+c)/2
        return (((s-a)+(s-b)+(s-c))**0.5)
    
    choice = int(input("Choice: 1-square, 2-circle, 3-regtangle, 4-traingle:"))
    if choice == 1:
       area=area_square()
    elif choice == 2:
        area=area_circle()
    elif choice == 3:
        area=area_rectangle()
    elif choice == 4:
        area=area_traingle()
    else:
        print("Not a valid choice")
    print('The area of shape is: ',area)


# functions that returns multiple values.
def amount():
    print('Using return for multiple values')
    Maharashtra = 5000
    Gujarat = 2500
    Delhi = 4000
    return Maharashtra,Gujarat,Delhi
    




if __name__ == "__main__":
    #area()
    # Calling the function
    a1,a2,a3 = amount()
    print(f'Sale in Maharashtra is: {a1}, Gujarath is: {a2}, Delhi is: {a3}')    

     