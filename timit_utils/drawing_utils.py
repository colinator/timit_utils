import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from typing import List, Optional, Sequence

ShowTimeTop = 0
ShowTimeBottom = 1
ShowTimeNone = 2

class Panel:
    def __init__(self, height_ratio: Optional[float], width_ratio: Optional[float], show_x_axis: bool = False) -> None:
        self._height_ratio = height_ratio
        self._width_ratio = width_ratio
        self.show_x_axis = show_x_axis

    @property
    def height_ratio(self):
        return self._height_ratio

    @property
    def width_ratio(self):
        return self._width_ratio

    def get_new_axis(self, figure: plt.Figure, gridspecsubplot: int) -> plt.axes:
        ax = figure.add_subplot(gridspecsubplot)
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')
        if self.show_x_axis:
            ax.xaxis.tick_top()
        else:
            plt.xticks([])
        plt.box(on=None)
        return ax

    def render(self, figure: plt.Figure, gridspecsubplot: int, is_top_level: bool = False) -> None:
        pass

class OneSquarePanel(Panel):
    def __init__(self) -> None:
        super().__init__(height_ratio=1, width_ratio=1, show_x_axis=False)

class BarsPanel(Panel):
    def __init__(self, bar_heights: List[float], show_x_axis: bool = False) -> None:
        super().__init__(height_ratio=1, width_ratio=1, show_x_axis=show_x_axis)
        self.bar_heights = bar_heights

    def render(self, figure: plt.Figure, gridspecsubplot: int, is_top_level: bool = False) -> None:
        ax = self.get_new_axis(figure, gridspecsubplot)
        plt.box(on=True)
        ax.spines['bottom'].set_color('#aaaaaa')
        ax.spines['top'].set_color('#00000000')
        ax.spines['right'].set_color('#00000000')
        ax.spines['left'].set_color('#aaaaaa')
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='grey')
        for i, h in enumerate(self.bar_heights):
            plt.bar(i+1, h, color='g' if i == 0 else 'r')

class AudioPanel(Panel):
    def __init__(self, audio: np.ndarray, show_x_axis: bool = False) -> None:
        super().__init__(height_ratio=3, width_ratio=10, show_x_axis=show_x_axis)
        self.audio = audio

    def render(self, figure: plt.Figure, gridspecsubplot: int, is_top_level: bool = False) -> None:
        ax = self.get_new_axis(figure, gridspecsubplot)
        plt.xlim(xmax = self.audio.shape[0])
        plt.plot(self.audio, 'limegreen', alpha=0.7, linewidth=0.3)

class SignalPanel(Panel):
    def __init__(self, signal: np.ndarray, show_x_axis: bool = False) -> None:
        super().__init__(height_ratio=1, width_ratio=10, show_x_axis=show_x_axis)
        self.signal = signal

    def render(self, figure: plt.Figure, gridspecsubplot: int, is_top_level: bool = False) -> None:
        ax = self.get_new_axis(figure, gridspecsubplot)
        maxy = np.max(self.signal)
        miny = np.min(self.signal)
        minyizzero = abs(miny) < 0.000000001
        plt.yticks([maxy] if minyizzero else [miny, maxy])
        plt.xlim(xmax = self.signal.shape[0])
        plt.plot(self.signal, 'limegreen', linewidth=0.7)

class SignalStringFiringPanel(Panel):
    def __init__(self, signal: np.ndarray, firings: np.ndarray, string_name: str, string_processes_df: pd.DataFrame = None, show_x_axis: bool = False) -> None:
        super().__init__(height_ratio=1.5, width_ratio=10, show_x_axis=show_x_axis)
        self.signal = signal
        self.firings = firings
        self.string_name = string_name
        self.string_processes_df = string_processes_df

    def render(self, figure: plt.Figure, gridspecsubplot: int, is_top_level: bool = False) -> None:
        ax = self.get_new_axis(figure, gridspecsubplot)
        maxy = np.max(self.signal)
        miny = np.min(self.signal)
        minyizzero = abs(miny) < 0.000000001
        plt.yticks([maxy] if minyizzero else [miny, maxy])
        plt.xlim(xmax = self.signal.shape[0])
        plt.plot(self.signal, 'limegreen', linewidth=0.4, alpha=0.7)

        vax = ax.twinx()
        plt.box(on=False)
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(self.string_name, color='blue')
        plt.xlim(xmax = self.signal.shape[0])
        #plt.ylim(ymin=-0.01, ymax=1.01)
        plt.yticks([])
        plt.plot(self.firings, 'blue', linewidth=1.0)

        if self.string_processes_df is not None:
            if type(self.string_processes_df) is pd.DataFrame:
                for word, sp in self.string_processes_df.iterrows():
                    rect = patches.Rectangle((sp[0], 0), sp[1]-sp[0], 30, color='cyan', alpha=0.25)
                    # Add the patch to the Axes
                    vax.add_patch(rect)
            else:
                sp = self.string_processes_df
                rect = patches.Rectangle((sp[0], 0), sp[1]-sp[0], 30, color='cyan', alpha=0.25)
                vax.add_patch(rect)


class FiringsPanel(Panel):
    def __init__(self, firings: np.ndarray, show_x_axis: bool = False) -> None:
        super().__init__(height_ratio=firings.shape[1]*0.02, width_ratio=10, show_x_axis=show_x_axis)
        self.firings = firings

    def render(self, figure: plt.Figure, gridspecsubplot: int, is_top_level: bool = False) -> None:
        ax = self.get_new_axis(figure, gridspecsubplot)
        ax.set_xlim([0, self.firings.shape[0]])
        ax.set_ylim([0, self.firings.shape[1]])
        n = self.firings.shape[1]
        for i in range(n):
            ff = [(x, y) for x, y in enumerate(self.firings[:,i]*(n-i+1)) if y != False]
            x = [k[0] for k in ff]
            y = [k[1] for k in ff]
            plt.plot(x, y, 'b.', markersize=3)

class StringProcessPanel(Panel):
    def __init__(self, strings_df: pd.DataFrame, color: str, max_time: int, show_x_axis: bool = False) -> None:
        super().__init__(height_ratio=1.5, width_ratio=10, show_x_axis=show_x_axis)
        self.strings_df = strings_df
        self.color = color
        self.max_time = max_time

    def render(self, figure: plt.Figure, gridspecsubplot: int, is_top_level: bool = False) -> None:
        ax = self.get_new_axis(figure, gridspecsubplot)
        plt.yticks([])
        ax.set_xlim([0, self.max_time])
        ax.set_ylim([-3, 3])
        for i, (string, start, end) in enumerate(zip(self.strings_df.index, self.strings_df['start'], self.strings_df['end'])):
            y = (1 if i % 2 == 0 else -1)
            rect = patches.Rectangle((start,y-1), end-start, 2, linewidth=0, facecolor=self.color, alpha=0.5)
            ax.add_patch(rect)
            plt.text(end, y, string, ha='left', va='center', color=self.color)

class WordsPanel(StringProcessPanel):
    def __init__(self, words_df: pd.DataFrame, max_time: int, show_x_axis: bool = False) -> None:
        super().__init__(strings_df=words_df, color='purple', max_time=max_time, show_x_axis=show_x_axis)

class PhonesPanel(StringProcessPanel):
    def __init__(self, phones_df: pd.DataFrame, max_time: int, show_x_axis: bool = False) -> None:
        super().__init__(strings_df=phones_df, color='blue', max_time=max_time, show_x_axis=show_x_axis)


class CompositePanel(Panel):

    def __init__(self, sub_panels: Sequence[Panel], is_vertical: bool = True) -> None:
        super().__init__(height_ratio=None, width_ratio=None)
        self.sub_panels = sub_panels
        self.is_vertical = is_vertical
        self.wspace = 0.1
        self.hspace = 0.1

    @property
    def height_ratio(self):
        return sum((panel.height_ratio for panel in self.sub_panels))

    @property
    def width_ratio(self):
        return sum((panel.width_ratio for panel in self.sub_panels))

    @property
    def width_ratio(self):
        return 1.0

    def render(self, figure: plt.Figure, gridspecsubplot: int = 0, is_top_level: bool = False) -> None:
        height_ratios = [panel.height_ratio for panel in self.sub_panels] if self.is_vertical else [ 1 ]
        width_ratios =  [ 1 ] if self.is_vertical else [panel.width_ratio for panel in self.sub_panels]
        v = len(height_ratios)
        h = len(width_ratios)

        if is_top_level:
            grid = gridspec.GridSpec(v, h, height_ratios=height_ratios, width_ratios=width_ratios, wspace=self.wspace, hspace=self.hspace)
        else:
            grid = gridspec.GridSpecFromSubplotSpec(v, h, subplot_spec=gridspecsubplot, height_ratios=height_ratios, width_ratios=width_ratios, wspace=self.wspace, hspace=self.hspace)

        for i, panel in enumerate(self.sub_panels):
            panel.render(figure, grid[i])

class VerticalPanel(CompositePanel):
    def __init__(self, sub_panels: Sequence[Panel]) -> None:
        super().__init__(sub_panels, is_vertical = True)

class HorizontalPanel(CompositePanel):
    def __init__(self, sub_panels: Sequence[Panel]) -> None:
        super().__init__(sub_panels, is_vertical = False)

class SignalsPanel(VerticalPanel):
    def __init__(self, signals: np.ndarray) -> None:
        super().__init__([SignalPanel(signals[:,i]) for i in range(signals.shape[1])])

class SignalFiringPerformancePanel(HorizontalPanel):
    def __init__(self, signal: np.ndarray, firings: np.ndarray, string_name: str, string_processes_df: pd.DataFrame = None, performance: np.ndarray = None, show_x_axis: bool = False) -> None:
        panel1 = SignalStringFiringPanel(signal, firings, string_name, string_processes_df, show_x_axis)
        panel2 = BarsPanel(performance, show_x_axis=False)
        super().__init__([panel1, panel2])

class SignalsStringsFiringsPerformancePanel(VerticalPanel):
    def __init__(self, signals: np.ndarray, firings: np.ndarray, performance_df: pd.DataFrame, strings_df: pd.DataFrame) -> None:
        panels = [SignalFiringPerformancePanel(
                    signal = signals[:,i],
                    firings = firings[:,i],
                    string_name = key,
                    string_processes_df = strings_df.loc[key],
                    performance = performance_df.loc[key],
                    show_x_axis = i == 0)
                    for i, key in enumerate(performance_df.index)]
        super().__init__(panels)
        self.wspace = 0.2
        self.hspace = 0.2

    @property
    def height_ratio(self):
        return super().height_ratio * 0.7

class Figure:
    def __init__(self, width:int, dpi:int, scale: float, panel: CompositePanel) -> None:
        self.width = width
        self.dpi = dpi
        self.scale = scale
        self.panel = panel

    def render(self):
        fig = plt.figure(num=None, figsize=(self.width, self.panel.height_ratio * self.scale), dpi=self.dpi)
        self.panel.render(fig, gridspecsubplot=0, is_top_level=True)


def DrawVerticalPanels(vertical_panels: Sequence[Panel],
                       width:int=20,
                       dpi:int=80,
                       scale:float=0.5) -> None:
    f = Figure(width=width, dpi=dpi, scale=scale, panel=VerticalPanel(vertical_panels))
    f.render()
