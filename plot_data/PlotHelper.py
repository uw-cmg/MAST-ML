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
from sklearn.metrics import mean_squared_error
from matplotlib import cm as cm

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
        #for grouping plots
        self.group_dict=None #data per group
        self.outlying_groups=list() #list of outlying groups to mark
        #end grouping plots
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        #Attributes below are set in code.
        self.numlines=0 #will be set in self.verify()
        self.smallfont = 0.85*self.fontsize
        matplotlib.rcParams.update({'font.size': self.fontsize})
        matplotlib.rcParams.update({'axes.unicode_minus': False})
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
        if not(isinstance(self.sizes, list) or isinstance(self.sizes, np.ndarray)):
            self.sizes=np.array(self.sizes.split(","),'float')
        else:
            self.sizes = np.array(self.sizes, 'float') #make sure they are floats
        if type(self.marklargest) is str:
            self.marklargest = np.array(self.marklargest.split(","),'int')
        elif (isinstance(self.marklargest, list) or isinstance(self.marklargest, np.ndarray)):
            self.marklargest = np.array(self.marklargest,'int')
        else:
            raise ValueError("marklargest %s could not be identified." % self.marklargest)
        if (self.mlabellist is None) or (len(self.mlabellist) == 0):
            self.mlabellist = np.copy(self.xdatalist)
        elif type(self.mlabellist) is list:
            pass
        else:
            raise ValueError("mlabellist %s could not be identified. Should be None or a nested list to match each data series and point." % self.mlabellist)
        return

    def sort_series(self, xvals, yvals, xerr, yerr, verbose=0):
        """Sort x and y according to x. 
        """
        raise NotImplementedError("Deprecating sort_series")
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

    def write_csv_data_section(self, csvname, colname, label="array"):
        if '\\' in csvname:
            csvname = csvname.replace("\\","\\\\") #escape windows backslashes
        section="""\
        \n
        df_%s = pd.read_csv('%s')
        %s = df_%s['%s']\n
        """ % (label, csvname, label, label, colname)
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
            if not(label[0] == "_"): #regular line
                return self.write_line_section(seriesobj, label, usecsv=True)
            else: #guideline or other non-data line
                return self.write_line_section(seriesobj, label, usecsv=False)
        elif type(seriesobj) == matplotlib.container.ErrorbarContainer:
            return self.write_errorbar_section(seriesobj, label)
        else:
            raise ValueError("No implemented matching object type for %s" % type(seriesobj))
        return

    def write_line_section(self, lineobj, label="Line", usecsv=True):
        [xdata, ydata] = lineobj.get_data()
        nospace_label = label.replace(" ","_").replace("-","_")
        xdata_label = "%s_x" % nospace_label
        ydata_label = "%s_y" % nospace_label
        if usecsv is True:
            #savecsv = os.path.join(self.save_path,"%s_data_%s.csv" % (self.plotlabel, nospace_label))
            savecsv = "%s_data_%s.csv" % (self.plotlabel, nospace_label)
            xsection = self.write_csv_data_section(savecsv, self.xlabel, xdata_label)
            ysection = self.write_csv_data_section(savecsv, self.ylabel, ydata_label)
        else:
            xsection = self.write_array_data_section(xdata, xdata_label)
            ysection = self.write_array_data_section(ydata, ydata_label)
        section="""\
        %s
        %s
        plt.plot(%s, %s, label='%s',
                color='%s', linestyle='%s', linewidth=%s,
                marker='%s', markersize=%s, markeredgewidth=%s,
                markeredgecolor='%s', markerfacecolor='%s')
        """ % (xsection, ysection, xdata_label, ydata_label, label,
        lineobj.get_color(), lineobj.get_linestyle(), lineobj.get_linewidth(),
        lineobj.get_marker(), lineobj.get_markersize(), lineobj.get_markeredgewidth(),
        lineobj.get_markeredgecolor(), lineobj.get_markerfacecolor())
        return section
    
    def write_array_data_section(self, single_array, label="array"):
        np.set_printoptions(threshold=np.inf)
        section="""\
        \n
        %s = %s\n
        """ % (label, repr(single_array))
        return section
    
    def write_errorbar_section(self, container, label="Line"):
        """
        """
        children = container.get_children()
        lineobj = children[0]
        [xdata, ydata] = lineobj.get_data()
        nospace_label = label.replace(" ","_").replace("-","_")
        xdata_label = "%s_x" % nospace_label
        ydata_label = "%s_y" % nospace_label
        #savecsv = os.path.join(self.save_path,"%s_data_%s.csv" % (self.plotlabel, nospace_label))
        savecsv = "%s_data_%s.csv" % (self.plotlabel, nospace_label)
        xsection = self.write_csv_data_section(savecsv, self.xlabel, xdata_label)
        ysection = self.write_csv_data_section(savecsv, self.ylabel, ydata_label)
        ect = 1
        if container.has_xerr:
            xerrnegsection = self.write_csv_data_section(savecsv, "xerr", "%s_neg_err" % xdata_label)
            xerrpossection = self.write_csv_data_section(savecsv, "xerr", "%s_pos_err" % xdata_label)
        if container.has_yerr:
            yerrnegsection = self.write_csv_data_section(savecsv, "yerr", "%s_neg_err" % ydata_label)
            yerrpossection = self.write_csv_data_section(savecsv, "yerr", "%s_pos_err" % ydata_label)
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
                color='%s', linestyle='%s', linewidth=%s,
                marker='%s', markersize=%s, markeredgewidth=%s,
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
        xticklabels=list()
        xtlabels = axisobj.get_xticklabels()
        for xtlabel in xtlabels:
            xticklabels.append(xtlabel.get_text())
        section="""\
        plt.xlabel('%s')
        plt.ylabel('%s')
        ax = plt.gca()
        #ax.margins(0.5,0.5) #set margins so points are not cut off
        #ax.set_xscale('log', nonposx='clip') #set log scale
        #ax.set_xlim([-10.0, 10.0]) #set limits on x axis. Similar for y axis.
        ax.set_xticks(%s)
        ax.set_xticklabels(%s, rotation=0.0)
        ax.set_yticks(%s)
        #
        ### Set additional dashed gridline at x=0
        #ax.set_xticks(np.array([0]),minor=True)
        #ax.xaxis.grid(which='minor', linestyle='--')
        #
        ### Add diagonal guideline
        #plt.plot([-10,10],[-10,10],'--', color='gray')
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
            axisobj.get_xticks().tolist(), xticklabels,
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


    def write_notebook(self, picklename="figure.pickle", nbfigname="notebook_figure.png", nbname="test.ipynb"):
        """Write a notebook for a single set of axes.
            Includes some help text for twinning a second y axis.
        """
        fig_handle = pickle.load(open(picklename,'rb'))
        codelist=list()
        codelist.append("""\
        #Loaded from %s
        """ % os.path.basename(picklename))
        codelist.append("""\
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from numpy import array
        from numpy import nan
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
        all_lines =fig_handle.axes[0].get_lines()
        for lineobj in all_lines:
            label = lineobj.get_label()
            if not(label[0] == "_"): #Labeled line, handled with legend above.
                pass
            elif label == "_nolegend_": #probably Container, handled above.
                pass
            else:
                codelist.append(self.write_series_section(lineobj, label))
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
        if not (lgd is None): #lgd is None if there are no labeled lines
            lgd.get_frame().set_alpha(0.5) #translucent legend
        """)
        codelist.append("""\
        plt.savefig("%s", bbox_inches="tight")
        plt.show()
        """ % nbfigname)
        code=""
        for codeitem in codelist:
            code = code + codeitem + "\n"
        nb = nbf.v4.new_notebook()
        nb['cells'] = [nbf.v4.new_code_cell(code)]
        with open(nbname, 'w', encoding="utf-8") as f:
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
        plt.plot([-10,10],[-1,1], linestyle='--', color='gray') #guideline
        plt.plot([-10,10],[0,0], linestyle=':', color='black', label="_zeroline") #guideline
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
            plt.annotate(note, xy=(1.05, notey), xycoords="axes fraction",
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
        plt.close()
        return

    def multiple_overlay(self):
        """Plot multiple xy overlay
        """
        fig_handle = plt.figure()
        ax1 = plt.gca()
        for nidx in range(0, self.numlines):
            label = self.labellist[nidx]
            xdata = self.xdatalist[nidx]
            ydata = self.ydatalist[nidx]
            xerr = self.xerrlist[nidx]
            yerr = self.yerrlist[nidx]
            #[xdata,ydata,xerr,yerr] = self.sort_series(xdata,ydata,xerr,yerr)
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
        if self.guideline == 1: #square the axes according to stepsize and draw line
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
        for note in self.notelist:
            plt.annotate(note, xy=(1.05, notey), xycoords="axes fraction",
                        fontsize=self.smallfont)
            notey = notey - notestep
        #ANNOTATIONS FOR LARGEST
        for nidx in range(0, self.numlines):
            marknum = self.marklargest[nidx]
            if marknum == 0: #no marking
                continue
            if int(self.guideline) == 0: #just rank on y
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
        if self.guideline:
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
        if len(self.xdatalist) == 1:
            if not (lgd1 is None):
                lgd1.set_visible(False) #do not show legend for single line
        plt.tight_layout()
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        plt.savefig(os.path.join(self.save_path, "%s" % self.plotlabel),
                    bbox_inches='tight')
        self.print_data() #print csv for every plot
        pname = os.path.join(self.save_path, "%s.pickle" % self.plotlabel)
        with open(pname,'wb') as pfile:
            pickle.dump(fig_handle, pfile) 
        plt.close()
        self.write_notebook(picklename=pname, 
            nbfigname = "%s_nb" % self.plotlabel,
            nbname = os.path.join(self.save_path, "%s.ipynb" % self.plotlabel))
        return

    def print_data(self, ycol_labels=None):
        for nidx in range(0, self.numlines):
            label = self.labellist[nidx]
            nospace_label = label.replace(" ","_").replace("-","_")
            savecsv = os.path.join(self.save_path,"%s_data_%s.csv" % (self.plotlabel, nospace_label))
            dataframe = pd.DataFrame() #index = np.arange(0, len(self.xdatalist[nidx])))
            if ycol_labels is None:
                ylabel = self.ylabel
            else:
                ylabel = ycol_labels[nidx]
            dataframe[self.xlabel] = self.xdatalist[nidx]
            if not(self.xerrlist[nidx] is None):
                dataframe['xerr'] = self.xerrlist[nidx]
            dataframe[ylabel] = self.ydatalist[nidx]
            if not (self.yerrlist[nidx] is None):
                dataframe['yerr'] = self.yerrlist[nidx]
            dataframe.to_csv(savecsv)
        return

    def plot_group_splits_with_outliers(self):
        """
        """
        self.xdatalist=list()
        self.ydatalist=list()
        self.labellist=list()
        self.xerrlist=list()
        self.yerrlist=list()
        otherxdata=list()
        otherxerrdata=list()
        otherydata=list()
        groups = list(self.group_dict.keys())
        groups.sort()
        show_rmse = 0
        for group in groups:
            if group in self.outlying_groups:
                self.xdatalist.append(self.group_dict[group]['xdata'])
                self.xerrlist.append(self.group_dict[group]['xerrdata'])
                self.ydatalist.append(self.group_dict[group]['ydata'])
                self.yerrlist.append(None)
                self.labellist.append(group)
                if 'rmse' in self.group_dict[group].keys():
                    show_rmse = 1 # if any RMSE shown, do RMSE for remaining
                    rmse = self.group_dict[group]['rmse']
                    self.notelist.append('{:<1}: {:.2f}'.format(group, rmse))
            else:
                otherxdata.extend(self.group_dict[group]['xdata'])
                otherxerrdata.extend(self.group_dict[group]['xerrdata'])
                otherydata.extend(self.group_dict[group]['ydata'])
        if len(otherxdata) > 0:
            self.xdatalist.insert(0,otherxdata) #prepend
            self.xerrlist.insert(0,otherxerrdata)
            self.ydatalist.insert(0,otherydata)
            self.yerrlist.insert(0,None)
            self.labellist.insert(0,"All others")
            if show_rmse == 1:
                all_other_rmse = np.sqrt(mean_squared_error(otherydata, otherxdata))
                self.notelist.append('{:<1}: {:.2f}'.format("All others", all_other_rmse))
        self.verify()
        self.multiple_overlay() 
        return
    def plot_rmse_vs_text(self):
        """Plot RMSE vs. text
            Takes single list entry in each of xdatalist and ydatalist.
        """
        fig_handle = plt.figure()
        ax1 = plt.gca()
        rms_list = np.array(self.ydatalist[0],'float') #verify type
        # graph rmse vs left-out group
        group_list = self.xdatalist[0]
        numeric_list = np.arange(0, len(group_list))
        skipticks = np.ceil(len(numeric_list)/8)
        xticks = np.arange(0, max(numeric_list) + 1, skipticks, dtype='int')
        xticklabels = list()
        for xtick in xticks:
            xticklabels.append(group_list[xtick])
        ax1.plot(numeric_list, rms_list, 
                        linestyle='None',
                        marker='o', markersize=5,
                        markeredgecolor='blue',
                        markerfacecolor='blue',
                        markeredgewidth=3,
                        label=self.labellist[0])
        #plot zero line
        ax1.plot((0, max(numeric_list)+1), (0, 0), ls="--", c=".3") 
        plt.xticks(xticks, xticklabels)
        ax1.set_xlabel(self.xlabel)
        ax1.set_ylabel(self.ylabel)
        notey = 0.88
        notestep = 0.07
        for note in self.notelist:
            plt.annotate(note, xy=(1.05, notey), xycoords="axes fraction",
                        fontsize=self.smallfont)
            notey = notey - notestep
        if self.marklargest is None:
            pass
        else:
            for largerms_index in np.argsort(rms_list)[-1*self.marklargest[0]:]:
                alabel = group_list[largerms_index]
                ax1.annotate(s = alabel,
                            xy = (numeric_list[largerms_index], 
                                    rms_list[largerms_index]),
                            fontsize=self.smallfont)
        loc1 = 'best'
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
        if len(self.xdatalist) == 1:
            if not (lgd1 is None):
                lgd1.set_visible(False) #do not show legend for single line
        plt.tight_layout()
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        plt.savefig(os.path.join(self.save_path, "%s" % self.plotlabel),
                    bbox_inches='tight')
        self.print_data() #print csv for every plot
        pname = os.path.join(self.save_path, "%s.pickle" % self.plotlabel)
        with open(pname,'wb') as pfile:
            pickle.dump(fig_handle, pfile) 
        plt.close()
        self.write_notebook(picklename=pname, 
            nbfigname = "%s_nb" % self.plotlabel,
            nbname = os.path.join(self.save_path, "%s.ipynb" % self.plotlabel))
        return

    def plot_2d_rmse_heatmap(self):
        """Plot 2d hex heatmap
        """
        fig_handle = plt.figure()
        xvals = self.xdatalist[0]
        yvals = self.ydatalist[0]
        rmses = self.ydatalist[1]
        plt.hexbin(xvals, yvals,
                    C = rmses, 
                    gridsize=15,
                    cmap = cm.plasma,
                    bins=None,
                    vmax = max(rmses))
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        cb = plt.colorbar()
        cb.set_label('RMSE')
        plt.savefig(os.path.join(self.save_path, "%s" % self.plotlabel),
                    bbox_inches='tight')
        self.print_data(ycol_labels=[self.ylabel, 'RMSE']) #print csv for every plot
        pname = os.path.join(self.save_path, "%s.pickle" % self.plotlabel)
        with open(pname,'wb') as pfile:
            pickle.dump(fig_handle, pfile) 
        plt.close()
        #No notebook support for hexbin types yet
        print("No jupyter notebook will be printed for this plot.")
        #self.write_notebook(picklename=pname, 
        #    nbfigname = "%s_nb" % self.plotlabel,
        #    nbname = os.path.join(self.save_path, "%s.ipynb" % self.plotlabel))
        return
    
    def plot_3d_rmse_heatmap(self):
        """Plot 3d rmse heatmap
        """
        from mpl_toolkits.mplot3d import Axes3D
        fig_handle = plt.figure()
        ax = plt.gca()
        ax = fig_handle.add_subplot(111, projection='3d')
        xvals = np.array(self.xdatalist[0])
        yvals = np.array(self.ydatalist[0])
        zvals = np.array(self.ydatalist[1])
        zlabel = self.labellist[1]
        rmses = np.array(self.ydatalist[2])
        scatter_series = ax.scatter(xvals, yvals, zvals,
                    marker='o',
                    c = rmses,
                    s = 20,
                    cmap = cm.plasma)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_zlabel(zlabel)
        cb = fig_handle.colorbar(scatter_series)
        cb.set_label('RMSE')
        plt.savefig(os.path.join(self.save_path, "%s" % self.plotlabel),
                    bbox_inches='tight')
        self.print_data(ycol_labels=[self.ylabel, zlabel, 'RMSE']) #print csv for every plot
        pname = os.path.join(self.save_path, "%s.pickle" % self.plotlabel)
        with open(pname,'wb') as pfile:
            pickle.dump(fig_handle, pfile) 
        plt.close()
        #No notebook support for hexbin types yet
        print("No jupyter notebook will be printed for this plot.")
        #self.write_notebook(picklename=pname, 
        #    nbfigname = "%s_nb" % self.plotlabel,
        #    nbname = os.path.join(self.save_path, "%s.ipynb" % self.plotlabel))
        return

        
