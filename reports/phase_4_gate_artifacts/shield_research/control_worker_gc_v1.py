"""Bounded cyclic-GC discipline for an already isolated control-worker child.

Importing this helper changes nothing. Refcounting remains active; only automatic
cyclic GC is disabled inside the context. This is not an execution-time proof.
"""
import gc
import os
from pathlib import Path
import time

MAX_SUPPORTED_STEPS=4500

class ControlStepBudgetExceeded(RuntimeError):pass
class ControlWorkerDisciplineError(RuntimeError):pass

def _count(value,name,*,allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if allow_zero else "positive"} integer')
    return value

def _stats():return [dict(row) for row in gc.get_stats()]
def _delta(after,before):
    return [{key:row[key]-old[key] for key in row} for row,old in zip(after,before)]
def _rss():
    try:
        fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
        return {'current_rss_kib':int(fields['VmRSS'].split()[0]),'process_peak_rss_kib':int(fields['VmHWM'].split()[0]),
                'scope':'Linux process RSS/current and lifetime high-water mark'}
    except (OSError,KeyError,ValueError):
        return {'current_rss_kib':None,'process_peak_rss_kib':None,'scope':'RSS unavailable'}

class ControlWorkerGC:
    """Use only inside a dedicated child, for at most1..4500 declared ticks.

    Preferred per-tick use: control_step() immediately before guard execution,
    physical_step_completed() after its physical word completes. For a separately
    frozen fixed-count loop, reserve_declared_steps() and report_declared_result()
    preserve that loop unchanged while making the different enforcement explicit.
    """
    _active_owner=None
    def __init__(self,*,max_steps):
        self.max_steps=_count(max_steps,'max_steps')
        if self.max_steps>MAX_SUPPORTED_STEPS:raise ValueError('maximum supported control horizon is4500 steps')
        self._entered=False;self._finished=False;self._mode=None
        self.reserved_steps=0;self.attempted_control_steps=0;self.completed_physical_words=0
        self.declared_result_reported=False;self.report=None
    def __enter__(self):
        if self._entered or self._finished or ControlWorkerGC._active_owner is not None:
            raise ControlWorkerDisciplineError('one non-reusable control GC context per worker at a time')
        self.entry_start=time.perf_counter()
        self.pid=os.getpid();self.parent_pid=os.getppid();self.prior_enabled=gc.isenabled()
        self.initial_counts=gc.get_count();self.initial_stats=_stats();self.rss_before=_rss();self.thresholds=gc.get_threshold()
        ControlWorkerGC._active_owner=self
        start=time.perf_counter()
        try:
            self.preflight_collected=gc.collect()
            self.preflight_elapsed_s=time.perf_counter()-start
            self.after_collect_stats=_stats();self.after_collect_counts=gc.get_count();self.rss_after_collect=_rss()
            gc.disable();self.control_initial_stats=_stats();self.control_start=time.perf_counter();self._entered=True
            return self
        except BaseException:
            gc.enable() if self.prior_enabled else gc.disable()
            ControlWorkerGC._active_owner=None
            raise
    def _check(self):
        if not self._entered or self._finished or os.getpid()!=self.pid:
            raise ControlWorkerDisciplineError('active same-process worker context required')
        if gc.isenabled():raise ControlWorkerDisciplineError('cyclic GC was re-enabled inside the control context')
    def _reserve(self,count):
        self._check();count=_count(count,'step count')
        if self.reserved_steps+count>self.max_steps:
            raise ControlStepBudgetExceeded('declared control-step budget exceeded')
        self.reserved_steps+=count
    def control_step(self,count=1):
        if self._mode not in (None,'per_tick'):raise ControlWorkerDisciplineError('cannot mix per-tick and frozen-loop counting')
        self._reserve(count);self._mode='per_tick';self.attempted_control_steps+=count
    def physical_step_completed(self,count=1):
        self._check();count=_count(count,'completed word count')
        if self._mode!='per_tick' or self.completed_physical_words+count>self.attempted_control_steps:
            raise ControlWorkerDisciplineError('physical completions cannot exceed attempted ticks')
        self.completed_physical_words+=count
    def reserve_declared_steps(self,count):
        if self._mode is not None:raise ControlWorkerDisciplineError('one frozen fixed-count loop per context')
        self._reserve(count);self._mode='sealed_declared_loop'
    def report_declared_result(self,*,attempted_steps,completed_words):
        self._check();attempted_steps=_count(attempted_steps,'attempted steps',allow_zero=True)
        completed_words=_count(completed_words,'completed words',allow_zero=True)
        if self._mode!='sealed_declared_loop' or self.declared_result_reported or not completed_words<=attempted_steps<=self.reserved_steps:
            raise ControlWorkerDisciplineError('frozen-loop result must fit the reserved step budget')
        self.attempted_control_steps=attempted_steps;self.completed_physical_words=completed_words;self.declared_result_reported=True
    def __exit__(self,kind,error,traceback):
        elapsed=time.perf_counter()-self.control_start
        try:
            end_stats=_stats();end_counts=gc.get_count();end_rss=_rss();enabled_before_restore=gc.isenabled()
        finally:
            gc.enable() if self.prior_enabled else gc.disable()
            self._finished=True;self._entered=False;ControlWorkerGC._active_owner=None
        delta=_delta(end_stats,self.control_initial_stats)
        self.report={'version':'control_worker_gc_v1','pid':self.pid,'parent_pid':self.parent_pid,
            'scope':'caller must launch this context in an isolated child; Python process-local GC only',
            'configured_maximum_steps':self.max_steps,'hard_supported_maximum_steps':MAX_SUPPORTED_STEPS,
            'counting_mode':self._mode,'reserved_steps':self.reserved_steps,
            'attempted_control_steps':None if self._mode=='sealed_declared_loop' and not self.declared_result_reported else self.attempted_control_steps,
            'completed_physical_words':None if self._mode=='sealed_declared_loop' and not self.declared_result_reported else self.completed_physical_words,'declared_result_reported':self.declared_result_reported,
            'counts_are_reported_not_per_tick_instrumented':self._mode=='sealed_declared_loop',
            'prior_gc_enabled':self.prior_enabled,'gc_enabled_inside_before_restore':enabled_before_restore,
            'restored_gc_enabled':gc.isenabled(),'prior_enabled_state_restored':gc.isenabled()==self.prior_enabled,
            'thresholds_before':self.thresholds,'thresholds_after':gc.get_threshold(),
            'preflight_collect_elapsed_s':self.preflight_elapsed_s,'preflight_collected_objects':self.preflight_collected,
            'control_context_elapsed_s':elapsed,'total_preflight_plus_control_elapsed_s':elapsed+(self.control_start-self.entry_start),
            'preflight_total_wall_elapsed_s':self.control_start-self.entry_start,
            'initial_gc_stats':self.initial_stats,'after_preflight_collect_gc_stats':self.after_collect_stats,
            'control_initial_gc_stats':self.control_initial_stats,'end_control_gc_stats':end_stats,'control_gc_stats_delta':delta,
            'collections_occurred_inside_control':any(row['collections'] for row in delta),
            'gc_count_before_preflight':self.initial_counts,'gc_count_after_preflight':self.after_collect_counts,'gc_count_end_control':end_counts,
            'rss_before_preflight':self.rss_before,'rss_after_preflight':self.rss_after_collect,'rss_end_control':end_rss,
            'exception_type':kind.__name__ if kind else None,'reference_counting_unchanged':True,
            'no_worst_case_execution_time_claim':True,'flight_certificate':False}
        return False
