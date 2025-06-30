# functions with arguments
# Program to determine whether two number are amicable.
def numbersAreAmicable(num):
    def sumoffact(num):
        sum=1
        for i in range(2,num):
            if num%i==0:
                sum=sum+i
        return sum
    # Accepting input from the user
    x,y=eval(input("Enter the two numbers: "))
    # Calling the function.
    val1 = sumoffact(x)
    val2 = sumoffact(y)

    # Check condition of amicable nos.through output returned from fucntion
    if (val1==y)&(val2==x):
        print("The numbers are amicable")
    else:
        print("The numbers are not amicable")





if __name__ == "__main__":
    main()
    # Accepting inp
   