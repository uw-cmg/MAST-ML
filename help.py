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
    #hmodule = __import__(hitem)
    strhelp = pydoc.render_doc(hitem)
    print(strhelp)
    #for mname in mnames:
    #    hexec = getattr(hmodule, mname)
    #    if callable(hexec):
    #        print(hexec.__doc__)
