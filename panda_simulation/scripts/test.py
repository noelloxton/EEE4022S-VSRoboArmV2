user_in = 0
while ((user_in < 0.3224) or (user_in >1.2724)): 
  try:
    user_in = input("Enter a Y-coordinate spawn position for the target (Please Enter in the range: 0.3224 <= y <= 1.2724 ): ")
    user_in = float(user_in)
  except ValueError:
    print("Error: that's not a valid input:(. Please try again")

print(user_in)
