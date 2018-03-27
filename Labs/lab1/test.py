def sum(s):
    s=str(s)
    s=s.split('+')
    return int(s[0])+int(s[1])
    
def main():
    s=input().strip()
    print (sum(s))
 
if __name__=="__main__":
    main()