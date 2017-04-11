#!/usr/bin/env python
##########
# Display documentation
##########
import pydoc

hlist=list()
hlist.append("FullFit")
hlist.append("KFoldCV")
hlist.append("LeaveOneOutCV")
hlist.append("LeaveOutGroupCV")
hlist.append("LeaveOutPercentCV")
hlist.append("ExtrapolateFullFit")
hlist.append("GAparamsearch")

for hitem in hlist:
    print("-------------------------")
    print("%s" % hitem)
    print(".........................")
    hmodule = __import__(hitem)
    hexec = getattr(hmodule, "execute")
    print(hexec.__doc__)
