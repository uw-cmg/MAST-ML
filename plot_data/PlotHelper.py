import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import heapq
import nbformat as nbf
import pickle
import inspect
class PlotHelper():
    """Plotting class
        Expects **kwargs dictionary.
        See __init__method for attributes.
    """
    def __init__(self, *args, **kwargs):
        #Attributes may be set by **kwargs
        self.xdatalist=list()
        self.xerrlist=list()
        self.ydatalist=list() 
        self.yerrlist=list()
        self.labellist=list()
        self.xlabel="X"
        self.ylabel="Y"
        self.save_path=""
        self.plotlabel="multiple_overlay"
        self.guideline=0
        self.notelist=list() 
        self.marklargest="0,0,0,0,0,0"
        self.mlabellist=None
        self.markers="o,o,s,d,^,v"
        self.linestyles="None,None,None,None,None,None"
        self.outlines="#8B0000,#00008B,#004400,#542788,#b35806,#252525"
        self.faces="red,blue,green,#6a51a3,orange,#bdbdbd"
        self.sizes="15,10,8,8,8,8"
        self.legendloc=None
        self.fontsize=18
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        #Attributes below are set in code.
        self.numlines=0 #will be set in self.verify()
        self.smallfont = 0.85*self.fontsize
        matplotlib.rcParams.update({'font.size': self.fontsize})
        self.verify()
        return

    def verify(self):
        self.guideline=int(self.guideline)
        self.numlines=len(self.xdatalist)
        if self.numlines > 6:
            raise ValueError("Only 6 lines supported.")
        if not(len(self.ydatalist) == self.numlines):
            raise ValueError("Number of y series does not match number of x series.")
        if not(len(self.labellist) == self.numlines):
            raise ValueError("Number of labels does not match number of x series.")
        if not(len(self.xerrlist) == self.numlines):
            raise ValueError("Number of x error data series does not match number of x series. Use python None as series entry for no error.")
        if not(len(self.yerrlist) == self.numlines):
            raise ValueError("Not enough y error data series does not match number of x series. Use python None as series entry for no error.")
        if not(type(self.faces) is list):
            self.faces = self.faces.split(",")
        if not(type(self.outlines) is list):
            self.outlines = self.outlines.split(",")
        if not(type(self.linestyles) is list):
            self.linestyles = self.linestyles.split(",")
        if not(type(self.markers) is list):
            self.markers = self.markers.split(",")
        if not(type(self.sizes) is list):
            self.sizes=np.array(self.sizes.split(","),'float')
        else:
            self.sizes = np.array(self.sizes, 'float') #make sure they are floats
        if type(self.marklargest) is str:
            self.marklargest = np.array(self.marklargest.split(","),'int')
        elif type(self.marklargest) is list:
            self.marklargest = np.array(self.marklargest,'int')
        else:
            raise ValueError("marklargest %s could not be identified." % self.marklargest)
        if self.mlabellist is None:
            self.mlabellist = np.copy(self.xdatalist)
        elif type(self.mlabellist) is str:
            self.mlabellist = self.mlabellist.split(",")
        elif type(self.mlabellist) is list:
            pass
        else:
            raise ValueError("mlabellist %s could not be identified." % self.mlabellist)
        return

    def sort_series(self, xvals, yvals, xerr, yerr, verbose=0):
        """Sort x and y according to x. 
        """
        arraylist = list()
        arraylist.append(xvals)
        arraylist.append(yvals)
        if xerr is None:
            dummy_xerr = np.zeros(len(xvals))
            arraylist.append(dummy_xerr)
        else:
            arraylist.append(xerr)
        if yerr is None:
            dummy_yerr = np.zeros(len(yvals)) #len(yvals) should also be len(xvals)
            arraylist.append(dummy_yerr)
        else:
            arraylist.append(yerr)
        try:
            combarr = np.array(arraylist,'float')
        except ValueError as ve:
            import traceback
            print("Error combining array")
            print("Shapes for arraylist xvals, yvals, xerr, yerr")
            for cidx in range(0,4):
                print(arraylist[cidx].shape)
            raise ValueError(traceback.print_exc())
        if verbose > 0:
            print("Original:")
            print(combarr)
        sortedarr = combarr[:,np.argsort(combarr[0])]
        if verbose > 0:
            print("Sorted:")
            print(sortedarr)
        xsorted = sortedarr[0,:]
        ysorted = sortedarr[1,:]
        if xerr is None:
            xerrsorted = None
        else:
            xerrsorted = sortedarr[2,:]
        if yerr is None:
            yerrsorted = None
        else:
            yerrsorted = sortedarr[3,:]
        return [xsorted, ysorted, xerrsorted, yerrsorted]

    def test_all(self):
        self.plot_single_test()
        #self.get_member_info()
        self.write_notebook()
        return

    def get_member_info(self, fname="figure.pickle"):
        fig_handle = pickle.load(open(fname,'rb'))
        [xdata, ydata] = fig_handle.axes[0].lines[0].get_data()
        print("Axes members")
        axes_members = inspect.getmembers(fig_handle.axes[0])
        for amemb in axes_members:
            print(amemb)
        [series, labels] = fig_handle.axes[0].get_legend_handles_labels()
        for sidx in range(0, len(series)):
            print("Series members series %i" % sidx)
            series_members = inspect.getmembers(series[sidx])
            for smemb in series_members:
                print(smemb)
        return

    def write_data_section(self, single_array, label="array"):
        section="""\
        \n
        %s = %s\n
        """ % (label, repr(single_array))
        return section
    
    def write_series_section(self, seriesobj, label="Line"):
        """
        Args:
            seriesobj <Matplotlib Line2D object or ErrorbarContainer object>: 
                Line object or container object, e.g. from
                fig_handle.axes[0].lines[<index number>]
                or from get_legend_handles_labels
            label <str>: Series label
        """
        if type(seriesobj) == matplotlib.lines.Line2D:
            return self.write_line_section(seriesobj, label)
        elif type(seriesobj) == matplotlib.container.ErrorbarContainer:
            return self.write_errorbar_section(seriesobj, label)
        else:
            raise ValueError("No implemented matching object type for %s" % type(seriesobj))
        return

    def write_line_section(self, lineobj, label="Line"):
        [xdata, ydata] = lineobj.get_data()
        xdata_label = "%s_x" % label
        ydata_label = "%s_y" % label
        xsection = self.write_data_section(xdata, xdata_label)
        ysection = self.write_data_section(ydata, ydata_label)
        section="""\
        %s
        %s
        plt.plot(%s, %s, label='%s',
                color='%s', linestyle='%s', linewidth='%s',
                marker='%s', markersize='%s', markeredgewidth='%s',
                markeredgecolor='%s', markerfacecolor='%s')
        """ % (xsection, ysection, xdata_label, ydata_label, label,
        lineobj.get_color(), lineobj.get_linestyle(), lineobj.get_linewidth(),
        lineobj.get_marker(), lineobj.get_markersize(), lineobj.get_markeredgewidth(),
        lineobj.get_markeredgecolor(), lineobj.get_markerfacecolor())
        return section
    
    def write_errorbar_section(self, container, label="Line"):
        """
        """
        children = container.get_children()
        lineobj = children[0]
        [xdata, ydata] = lineobj.get_data()
        xdata_label = "%s_x" % label
        ydata_label = "%s_y" % label
        xsection = self.write_data_section(xdata, xdata_label)
        ysection = self.write_data_section(ydata, ydata_label)
        ect = 1
        if container.has_xerr:
            xerrnegobj = children[ect]
            [xerrnegdata, ydummy]=xerrnegobj.get_data()
            ect = ect + 1
            xerrposobj = children[ect]
            [xerrposdata, ydummy]=xerrposobj.get_data()
            ect = ect + 1
            xerrneg = xdata - xerrnegdata
            xerrpos = xerrposdata - xdata
            xerrnegsection = self.write_data_section(xerrneg, "%s_neg_err" % xdata_label)
            xerrpossection = self.write_data_section(xerrpos, "%s_pos_err" % xdata_label)
        if container.has_yerr:
            yerrnegobj = children[ect]
            [xdummy, yerrnegdata]=yerrnegobj.get_data()
            ect = ect + 1
            yerrposobj = children[ect]
            [xdummy, yerrposdata]=yerrposobj.get_data()
            ect = ect + 1
            yerrneg = ydata - yerrnegdata
            yerrpos = yerrposdata - ydata
            yerrnegsection = self.write_data_section(yerrneg, "%s_neg_err" % ydata_label)
            yerrpossection = self.write_data_section(yerrpos, "%s_pos_err" % ydata_label)
        mainsection="(_, caps, _)=plt.errorbar(%s, %s, label='%s'," % (xdata_label, ydata_label, label)
        errdatasection=""
        xerrline="#"
        yerrline="#"
        if container.has_xerr:
            errdatasection = errdatasection + xerrnegsection + xerrpossection
            xerrline =  "        xerr=(%s_neg_err,%s_pos_err)," % (xdata_label, xdata_label)
        if container.has_yerr:
            errdatasection = errdatasection + yerrnegsection + yerrpossection
            yerrline = "        yerr=(%s_neg_err,%s_pos_err)," % (ydata_label, ydata_label)
        customsection="""\
                color='%s', linestyle='%s', linewidth='%s',
                marker='%s', markersize='%s', markeredgewidth='%s',
                markeredgecolor='%s', markerfacecolor='%s')
        """ % (lineobj.get_color(), lineobj.get_linestyle(), lineobj.get_linewidth(),
        lineobj.get_marker(), lineobj.get_markersize(), lineobj.get_markeredgewidth(),
        lineobj.get_markeredgecolor(), lineobj.get_markerfacecolor())
        capssection="""\n
        for cap in caps:
            cap.set_color('%s')
            cap.set_markeredgewidth('%s')
        """ % (lineobj.get_color(), lineobj.get_markeredgewidth())
        section="""\
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        """ % (xsection, ysection, errdatasection, mainsection, xerrline, yerrline, customsection, capssection)
        return section

    def write_axis_section(self, axisobj):
        axistext=""
        section="""\
        plt.xlabel('%s')
        plt.ylabel('%s')
        ax = plt.gca()
        #ax.margins(0.5,0.5) #set margins so points are not cut off
        #ax.set_xscale('log', nonposx='clip') #set log scale
        #ax.set_xlim([-10.0, 10.0]) #set limits on x axis. Similar for y axis.
        ax.set_xticks(%s)
        #ax.set_xticklabels(["a","b","c","d","e"],rotation=90) #set tick labels
        ax.set_yticks(%s)
        #
        ### Set additional dashed gridline at x=0
        #ax.set_xticks(np.array([0]),minor=True)
        #ax.xaxis.grid(which='minor', linestyle='--')
        #
        ### Do not use scientific ticks with an automatic multiplier
        #y_formatter = matplotlib.ticker.ScalarFormatter()
        #y_formatter.set_useOffset(False)
        #y_formatter.set_scientific(False)
        #ax.yaxis.set_major_formatter(y_formatter)
        #
        ### Set xticks as times
        #import time
        #my_xticks = ax.get_xticks()
        #adjusted_xticks = list()
        #for tidx in range(0, len(my_xticks)):
        #    mytick = time.strftime("format string", time.localtime(my_xticks[tidx]))
        #               #see time.strftime for format strings to use
        #    adjusted_xticks.append(mytick)
        #    ax.set_xticklabels(adjusted_xticks, rotation=90.0)
        #
        ### Use a second y axis with the same x axis
        #ax2 = ax.twinx()
        #ax2.set_ylabel("second label")
        #ax2.plot(...) #line on second axis, for example, cut a plt.plot section
        #              #from above and paste it here, making it ax2.plot
        #ax2.errorbar(...) #line with errorbar on second axis, for example, cut
        #                  # a plt.errorbar section from above, making it
        #                  # (_, caps, _)=ax2.errorbar(...)
        #ax2.legend() #legend for entries on second axis. 
        """ % (axisobj.get_xlabel(), axisobj.get_ylabel(),
            axisobj.get_xticks().tolist(),
            axisobj.get_yticks().tolist())
        return section

    def write_annotation_section(self, axisobj):
        annotations=""
        for child in axisobj.get_children():
            if isinstance(child, matplotlib.text.Annotation):
                annotations=annotations+"""\
                \n        plt.annotate("%s", 
                    xy=%s, xycoords='%s',
                    fontsize=smallfont)\n
                """ % (child.get_text(), child.get_position(), child.xycoords)
        section="""\
        \n%s
        """ % annotations 
        return section

    def write_notebook(self, fname="figure.pickle", savename="notebook_figure.png"):
        """Write a notebook for a single set of axes.
            Includes some help text for twinning a second y axis.
        """
        fig_handle = pickle.load(open(fname,'rb'))
        codelist=list()
        codelist.append("""\
        #Loaded from %s
        """ % fname)
        codelist.append("""\
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        from numpy import array
        matplotlib.rcParams.update({'font.size': 18})
        smallfont = 0.85*matplotlib.rcParams['font.size']
        plt.figure(figsize=(8,6)) #size in inches
        """)
        [series, labels] = fig_handle.axes[0].get_legend_handles_labels()
        #([<matplotlib.lines.Line2D object at 0x10ea187f0>, <Container object of 3 artists>], ['sine', 'cosine'])
        for sidx in range(0, len(series)):
            seriesobj = series[sidx]
            label = labels[sidx]
            codelist.append(self.write_series_section(seriesobj, label))
        axisobj = fig_handle.axes[0]
        codelist.append(self.write_axis_section(axisobj))
        codelist.append(self.write_annotation_section(axisobj))
        codelist.append("""\
        lgd = ax.legend(loc='lower center', #location
                        ncol=2, #number of columns
                        numpoints=1, #number of points
                        bbox_to_anchor=(0.5,1), #anchor against the figure axes; this anchor pins the lower center point to x=half axis and y=full axis
                        fontsize=smallfont,
                        )
        lgd.get_frame().set_alpha(0.5) #translucent legend
        """)
        codelist.append("""\
        plt.savefig("%s", bbox_inches="tight")
        plt.show()
        """ % savename)
        code=""
        for codeitem in codelist:
            code = code + codeitem + "\n"
        nb = nbf.v4.new_notebook()
        nb['cells'] = [nbf.v4.new_code_cell(code)]
        fname = 'test.ipynb'
        with open(fname, 'w') as f:
            nbf.write(nb, f)
        return

    def plot_single_test(self):
        """Testing single plot
            2 data lines, one with error bar
            Markers and symbols
            Legend
        """
        smallfont = 0.85*matplotlib.rcParams['font.size']
        fig_handle = plt.figure()
        xvals = np.arange(-10,10.5,0.5)
        yvals = np.sin(xvals)
        plt.plot(xvals, yvals, 'b-',label="sine")
        xvals2 = np.arange(-5,5,1)
        yvals2 = np.cos(xvals2)
        yerr2 = np.arange(-0.5,0.5,0.1)
        xerrneg = 1.2*np.ones(10)
        xerrpos = np.zeros(10)
        plt.errorbar(xvals2, yvals2, yerr=yerr2, xerr=(xerrneg,xerrpos),
                        color='r',
                        linestyle=':',
                        linewidth=3,
                        marker='s',
                        markeredgewidth=2,
                        markersize=15,
                        markeredgecolor='darkgreen',
                        markerfacecolor='green',
                        label="cosine")
        plt.errorbar(xvals2-3.0, yvals2+0.5, 
                        yerr=(0.2*np.ones(10), 0.5*np.ones(10)), 
                        color='gray',
                        linestyle='--',
                        linewidth=2,
                        marker='o',
                        markeredgewidth=1,
                        markersize=10,
                        markeredgecolor='black',
                        markerfacecolor='blue',
                        label="cosine_series2")
        plt.xlabel('Number')
        plt.ylabel('Function value')
        notelist=list()
        notelist.append("Annotations:")
        notelist.append("  sine-type curves")
        notey = 0.88
        notestep=0.07
        for note in notelist:
            plt.annotate(note, xy=(0.05, notey), xycoords="axes fraction",
                    fontsize=smallfont)
            notey = notey - notestep
        for midx in [7,8,9]:
            plt.annotate("%3.3f" % yvals2[midx], 
                    xy=(xvals2[midx], yvals2[midx]),
                    horizontalalignment = "left",
                    verticalalignment = "bottom",
                    fontsize=smallfont)
        plt.legend()
        plt.savefig("figure.png")
        with open('figure.pickle','wb') as pfile:
            pickle.dump(fig_handle, pfile) 
        return

    def multiple_overlay(self):
        """Plot multiple xy overlay
        """
        #PLOTTING
        plt.figure()
        fig, ax1 = plt.subplots()
        for nidx in range(0, self.numlines):
            label = self.labellist[nidx]
            xdata = self.xdatalist[nidx]
            ydata = self.ydatalist[nidx]
            xerr = self.xerrlist[nidx]
            yerr = self.yerrlist[nidx]
            [xdata,ydata,xerr,yerr] = self.sort_series(xdata,ydata,xerr,yerr)
            (_, caps, _) = ax1.errorbar(xdata, ydata,
                xerr=xerr,
                yerr=yerr,
                label=label,
                linewidth=2,
                linestyle = self.linestyles[nidx], color=self.outlines[nidx],
                markeredgewidth=2, markeredgecolor=self.outlines[nidx],
                markerfacecolor=self.faces[nidx] , marker=self.markers[nidx],
                markersize=self.sizes[nidx])
            for cap in caps:
                cap.set_color(self.outlines[nidx])
                cap.set_markeredgewidth(2)
        #AXIS LABELS
        ax1.set_ylabel(self.ylabel)
        plt.xlabel(self.xlabel)
        #X and Y-AXIS RANGES FOR SQUARE PLOT
        if guideline == 1: #square the axes according to stepsize and draw line
            [minx,maxx] = ax1.get_xlim()
            [miny,maxy] = ax1.get_ylim()
            gmax = max(maxx, maxy)
            gmin = min(minx, miny)
            ax1.set_xlim([gmin,gmax]) #set both X and Y limits
            ax1.set_ylim([gmin,gmax])
            plt.plot((gmin, gmax), (gmin, gmax), ls="--", c=".3")
        #MARGINS
        plt.margins(0.05)
        #ANNOTATIONS
        notey = 0.88
        notestep = 0.07
        for note in notelist:
            plt.annotate(note, xy=(0.05, notey), xycoords="axes fraction",
                        fontsize=self.smallfont)
            notey = notey - notestep
        #ANNOTATIONS FOR LARGEST
        for nidx in range(0, numlines):
            marknum = self.marklargest[nidx]
            if marknum == 0: #no marking
                continue
            if int(guideline) == 0: #just rank on y
                torank = self.ydatalist[nidx]
            else: #rank on distance from x-y guideline
                torank = np.abs(self.ydatalist[nidx] - self.xdatalist[nidx])
            meantorank = np.nanmean(torank) #get rid of NaN's, esp. on CV test
            mynans = np.isnan(torank)
            torank[mynans] = meantorank #NaN's become means, so not high or low
            maxidxlist = heapq.nlargest(marknum, range(len(torank)), 
                            key=lambda x: torank[x])
            print(maxidxlist)
            for midx in maxidxlist:
                mxval = self.mlabellist[nidx][midx]
                try:
                    mxval = "%3.0f" % mxval
                except TypeError:
                    pass
                plt.annotate("%s" % mxval, 
                        xy=(self.xdatalist[nidx][midx],
                        self.ydatalist[nidx][midx]),
                        horizontalalignment = "left",
                        verticalalignment = "bottom",
                        fontsize=self.smallfont)
        #LEGEND
        if guideline:
            loc1 = "lower right"
        else:
            loc1 = "best"
        if not(self.legendloc is None):
            loc1 = self.legendloc
        lgd1=ax1.legend(loc = loc1, 
                    fontsize=self.smallfont, 
                    numpoints=1,
                    fancybox=True) 
        try:
            lgd1.get_frame().set_alpha(0.5) #translucent legend!
        except AttributeError: # no labeled lines
            pass
        plt.tight_layout()
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        plt.savefig(os.path.join(self.save_path, "%s" % self.plotlabel), 
                    dpi=200, bbox_inches='tight')
        plt.close()
        #PRINT DATA
        for nidx in range(0, numlines):
            label = labellist[nidx]
            nospace_label = label.replace(" ","_")
            savecsv = os.path.join(save_path,"data_%s.csv" % nospace_label)
            dataframe = pd.DataFrame(index = np.arange(0, len(xdatalist[nidx])))
            dataframe[xlabel] = xdatalist[nidx]
            if not(xerr is None):
                dataframe['xerr'] = xerrlist[nidx]
            dataframe[ylabel] = ydatalist[nidx]
            if not (yerr is None):
                dataframe['yerr'] = yerrlist[nidx]
            dataframe.to_csv(savecsv)
        return
