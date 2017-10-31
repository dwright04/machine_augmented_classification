import numpy as np

np.random.seed(1337)

import matplotlib.pyplot as plt

from math import factorial

def experiment1b(m, V, L):

  retired = {}
  subjects = [x for x in range(m)] * L # m x L classifications needed
  order = np.random.permutation(len(subjects))
  subjects = np.array(subjects)[order]
  time_counter = 1
  for i in range(0,len(subjects),V):
    start = i
    stop = i+V
    classified = subjects[start:stop]
    for s in classified:
      retired[s] = {'time to retirement':time_counter}
    time_counter += 1

  return retired

def meanTimeToRetirement(retired):
  time_to_retirement = []
  for key in retired.keys():
    time_to_retirement.append(retired[key]['time to retirement'])
  return np.mean(time_to_retirement)

def nChoosek(n, k):
  return factorial(n) / (factorial(k)*factorial(n-k))

def main():
  """
  m = 1000  # number of subjects
  V = 10      # number of volunteers
  L = 10       # retirement limit
  
  retired = experiment1b(m, V, L)

  bins = [x for x in np.arange(0,(m)*1.1, 100)]
  time_to_retirement = []
  for key in retired.keys():
    time_to_retirement.append(retired[key]['time to retirement'])

  counts, _, _ = plt.hist(time_to_retirement, color='#A0D2DB', \
    bins=bins, label='m/V=%d'%(m/V), zorder=1000)
  """
  """
  m = 1000000   # number of subjects
  V = 100      # number of volunteers
  L = 10       # retirement limit
  
  retired = experiment1b(m, V, L)

  bins = [x for x in np.arange(0,(m/L)*1.1, 100)]
  time_to_retirement = []
  for key in retired.keys():
    time_to_retirement.append(retired[key]['time to retirement'])

  counts, _, _ = plt.hist(time_to_retirement, color='#F0A202', \
    bins=bins, label='m/V=%d'%(m/V))
  #plt.plot([(m*L)/V,(m*L)/V],[0,m*1.1],'k--')
  #plt.plot([np.mean(time_to_retirement),np.mean(time_to_retirement)],\
  #  [0,m*1.1],'k--')
  #plt.text(np.mean(time_to_retirement)-5000, m*1.11, '%d'%np.mean(time_to_retirement))
  
  m = 1000000   # number of subjects
  V = 1000      # number of volunteers
  L = 10        # retirement limit
  
  retired = experiment1b(m, V, L)

  time_to_retirement = []
  for key in retired.keys():
    time_to_retirement.append(retired[key]['time to retirement'])

  counts, _, _ = plt.hist(time_to_retirement, color='#B8336A', \
    bins=bins, label='m/V=%d'%(m/V))
  #plt.plot([(m*L)/V,(m*L)/V],[0,m*1.1],'k--')
  #plt.plot([np.mean(time_to_retirement),np.mean(time_to_retirement)],\
  #  [0,m*1.1],'k--')
  #plt.text(np.mean(time_to_retirement)-5000, m*1.11, '%d'%np.mean(time_to_retirement))
  """
  """
  m = 1000000   # number of subjects
  V = 10000     # number of volunteers
  L = 10        # retirement limit
  
  retired = experiment1b(m, V, L)

  time_to_retirement = []
  for key in retired.keys():
    time_to_retirement.append(retired[key]['time to retirement'])

  counts, _, _ = plt.hist(time_to_retirement, color='#726DA8', \
    bins=bins, label='m/V=%.2lf'%(m/V))
  plt.plot([(m*L)/V,(m*L)/V],[0,m*1.1],'k--')
  """
  
  #plt.yscale('log')
  #plt.ylim(ymax=m*1.1)
  #plt.legend()
  #plt.show()
  
  """
  m = 100.   # number of subjects
  V = 2.      # number of volunteers
  L = 10.       # retirement limit

  #time  = np.arange(1,(m/V)+1)
  time  = np.arange(1,50)
  Ps = [(L*V)/m]
  Pnots = [1-Ps[-1]]
  p = [0]
  n = 0
  for t in time:
    P = 1
    if t < L:
      for i in np.arange(1,t+1):
        #print(i)
        P *= ((L-i)*V) / float((m-t*V))
    elif t >= L:
      for i in np.arange(1,L+1):
        P *= ((L-i)*V) / float(m-t*V)
    #print(P)
    #p.append(P)
    Ps.append((L*V/(m-t))*P)
    #Pnots.append(1-Ps[-1])
  plt.plot([0]+list(time), Ps)
  #plt.plot([0]+list(time), Pnots)
  #plt.plot([0]+list(time), p)
  plt.ylim(0,1)
  plt.xlim(0,m-2*10+1)
  plt.show()
  """
  """
  m=1000
  L=10
  V=10

  time = list(np.arange(1,m+1))
  p_a = []
  p_b = []
  p_c = []
  p = [1/(L*m)]
  for t in time:
    a = (L - t) / ((L * m) - V*t)
    b = 0
    if a < 0:
      a = 0
    if t <= L:
      for i in range(2,t+1):
        b += i*((L-(i-1))/(L*m-t*V))
    elif t > L:
      for i in range(2,L+1):
        b += i*((L-(i-1))/(L*m-t*V))
    try:
      c = L / ((L * m) - V*t)
    except ZeroDivisionError:
      c = 1.0
    p_a.append(a)
    p_b.append(b)
    p_c.append(c)
    p.append(a+b+c)

  plt.plot(time, p_a)
  plt.plot(time, p_b)
  plt.plot(time, p_c)
  plt.plot([0]+time, p, label='P')
  plt.xlabel(r'time [arbitrary units]')
  plt.ylabel(r'$P(s_i)$')
  plt.ylim(0,1)
  plt.legend()
  plt.show()
  """

  M = [np.power(10, x) for x in range(3,7)]
  #L = [np.power(10, x) for x in range(1,3)]
  #V = [np.power(10, x) for x in range(1,5)]
  V = 1000
  L = 10
  times = []
  for m in M:
    retired = experiment1b(m, V, L)
    times.append(meanTimeToRetirement(retired))
  plt.plot(M,times,'o',color='#F0A202',label='mean time to retirement')
  plt.plot(M,(L*np.array(M)/V),'k--',label='L*m/V')
  plt.ylabel('time to retirement [arbitrary units]')
  plt.xlabel('m')
  #plt.yscale('log')
  plt.xscale('log')
  plt.legend(loc='upper left')
  #plt.show()
  plt.savefig('plots/experiment1b.png')
  plt.savefig('plots/experiment1b.pdf')

if __name__ == '__main__':
  main()
