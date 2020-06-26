import sys

file=sys.argv[1]
input = open(file, 'r')

lines=input.readlines();


lines=lines[:-10]
#print(lines)
cnt=0
sum=0.0
for x in lines:
    x=x.split('\n')[0]
    cnt=cnt+1
    sum=sum+float(x)
    
print("avg_power=",sum/cnt)

