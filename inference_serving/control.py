def readWait(p):
    # print("waiting astra-sim")
    out = [""]
    while out[-1] != "Waiting\n" and out[-1] != "Checking Non-Exited Systems ...\n":
        out.append(p.stdout.readline())
        p.stdout.flush()
    # for i in out:
    #     if i != "Waiting\n":
    #         print(i, end='')
    return out

def checkEnd(p):
    out = ["",""]
    while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
        out.append(p.stdout.readline())
        p.stdout.flush()
    for i in out[4:]:
        print(i, end='')
    return out

def writeFlush(p, input):
    p.stdin.write(input+'\n')
    p.stdin.flush()
    return

def parseOutput(output):
    if 'cycles' in output:
        sp = output.split()
        sys = int(sp[0].split('[')[1].split(']')[0])
        id = int(sp[2])
        cycle = int(sp[4])

        return {'sys':sys,'id':id,'cycle':cycle}
    return