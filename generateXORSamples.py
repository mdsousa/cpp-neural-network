import random
flip = random.choice([True,False])
numSamples = 10000
for i in range(numSamples):
   if i%2 == 0:
      if flip:
         print("{1, 0, 1}", end="")
      else:
         print("{0, 1, 1}", end="")
   else:
      if flip:
         print("{1, 1, 0}", end="")
      else:
         print("{0, 0, 0}", end="")
   if i < 99:
      print(",")
   flip = random.choice([True, False])
　