"""GC settings restore in this isolated test process; no model/physics changes."""
import gc
import unittest
from control_worker_gc_v1 import ControlWorkerGC,ControlStepBudgetExceeded,ControlWorkerDisciplineError

class WorkerGCTests(unittest.TestCase):
    def test_success_collects_before_control_restores_and_counts(self):
        before=gc.isenabled();threshold=gc.get_threshold()
        with ControlWorkerGC(max_steps=1000) as worker:
            self.assertFalse(gc.isenabled());worker.control_step();worker.physical_step_completed()
        self.assertEqual(gc.isenabled(),before);self.assertEqual(gc.get_threshold(),threshold)
        self.assertEqual(worker.report['attempted_control_steps'],1)
        self.assertEqual(worker.report['completed_physical_words'],1)
        self.assertFalse(worker.report['collections_occurred_inside_control'])
        self.assertTrue(worker.report['prior_enabled_state_restored'])
    def test_exception_restores(self):
        before=gc.isenabled()
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with ControlWorkerGC(max_steps=4500) as worker:
                worker.control_step();raise RuntimeError('deliberate')
        self.assertEqual(gc.isenabled(),before);self.assertEqual(worker.report['exception_type'],'RuntimeError')
        self.assertEqual(worker.report['completed_physical_words'],0)
    def test_initially_disabled_restores_disabled(self):
        before=gc.isenabled();gc.disable()
        try:
            with ControlWorkerGC(max_steps=1) as worker:worker.control_step()
            self.assertFalse(gc.isenabled());self.assertFalse(worker.report['prior_gc_enabled'])
        finally:gc.enable() if before else gc.disable()
    def test_initially_disabled_exception_stays_disabled(self):
        before=gc.isenabled();gc.disable()
        try:
            with self.assertRaises(RuntimeError):
                with ControlWorkerGC(max_steps=1) as worker:raise RuntimeError('failure')
            self.assertFalse(gc.isenabled());self.assertTrue(worker.report['prior_enabled_state_restored'])
        finally:gc.enable() if before else gc.disable()
    def test_negative_bool_zero_and_excessive_counts_reject(self):
        for bad in (-1,0,True,1.,4501):
            with self.subTest(bad=bad),self.assertRaises(ValueError):ControlWorkerGC(max_steps=bad)
        with ControlWorkerGC(max_steps=1) as worker:
            for bad in (-1,0,True):
                with self.assertRaises(ValueError):worker.control_step(bad)
            worker.control_step()
            with self.assertRaises(ControlStepBudgetExceeded):worker.control_step()
            worker.physical_step_completed()
            with self.assertRaises(ControlWorkerDisciplineError):worker.physical_step_completed()
    def test_declared_frozen_loop_budget_and_mode_separation(self):
        with ControlWorkerGC(max_steps=1000) as worker:
            worker.reserve_declared_steps(1000)
            with self.assertRaises(ControlWorkerDisciplineError):worker.control_step()
            worker.report_declared_result(attempted_steps=1000,completed_words=1000)
        self.assertEqual(worker.report['reserved_steps'],1000)
        self.assertTrue(worker.report['counts_are_reported_not_per_tick_instrumented'])
        with ControlWorkerGC(max_steps=1000) as worker:
            with self.assertRaises(ControlStepBudgetExceeded):worker.reserve_declared_steps(1001)
    def test_reenable_and_explicit_collection_are_visible(self):
        before=gc.isenabled()
        with self.assertRaises(ControlWorkerDisciplineError):
            with ControlWorkerGC(max_steps=1) as worker:
                gc.enable();worker.control_step()
        self.assertEqual(gc.isenabled(),before)
        with ControlWorkerGC(max_steps=1) as worker:gc.collect()
        self.assertTrue(worker.report['collections_occurred_inside_control'])
    def test_no_nested_or_reused_context(self):
        worker=ControlWorkerGC(max_steps=1)
        with worker:
            with self.assertRaises(ControlWorkerDisciplineError):
                with ControlWorkerGC(max_steps=1):pass
        with self.assertRaises(ControlWorkerDisciplineError):worker.__enter__()

if __name__=='__main__':unittest.main(verbosity=2)
