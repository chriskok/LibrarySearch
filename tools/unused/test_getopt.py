import sys, getopt
opts, args = getopt.getopt(sys.argv[1:], '', ['cam=', 'file='])
opts = dict(opts)
print(opts)
cam_number = opts.get('--cam', 0)
file = opts.get('--file', 'helloha')

print(file)
