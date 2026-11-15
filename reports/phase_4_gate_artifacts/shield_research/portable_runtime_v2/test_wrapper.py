"""Small nonflight tests; no Torch import, simulator step or timing benchmark."""
import importlib.util
from pathlib import Path
import sys,tempfile,unittest,json,gc
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_component as wrapper


def row(case=0,steps=3,timing=True):
    return {'case':case,'requested_steps':steps,'completed_steps':steps,
            'functional_behavior_passed':True,'timing_gate_passed':timing,
            'trace':[{'tick':i} for i in range(steps)],'error':None,
            'guard_total_latency_s':{'count':steps}}


def load_gc():
    spec=importlib.util.spec_from_file_location('test_control_gc',HERE.parent/'control_worker_gc_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class WrapperTests(unittest.TestCase):
    def test_real_gc_context_with_frozen_loop_stub(self):
        module=load_gc();prior=gc.isenabled();calls=[]
        def run(**kwargs):
            calls.append(kwargs);self.assertFalse(gc.isenabled())
            result={'cases':[row(case=2)],'source_hashes_stable':True,'run_valid':True}
            Path(kwargs['output']).write_text(json.dumps(result));return result
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'raw.json'
            raw,report=wrapper.execute_wrapped(SimpleNamespace(run=run),module.ControlWorkerGC,
                                               case=2,steps=3,raw_output=output)
            self.assertEqual(calls,[{'cases':1,'start_case':2,'steps':3,'output':str(output)}])
            self.assertEqual(json.loads(output.read_text()),raw)
            self.assertEqual(report['attempted_control_steps'],3)
            self.assertEqual(report['completed_physical_words'],3)
            self.assertEqual(report['counting_mode'],'sealed_declared_loop')
            self.assertTrue(report['prior_enabled_state_restored'])
            self.assertFalse(report['collections_occurred_inside_control'])
            with self.assertRaises(FileExistsError):wrapper.execute_wrapped(
                SimpleNamespace(run=run),module.ControlWorkerGC,case=2,steps=3,raw_output=output)
        self.assertEqual(gc.isenabled(),prior)

    def test_partial_failure_tick_count_is_not_double_counted(self):
        module=load_gc();record=row();record.update(completed_steps=2,trace=[{'tick':0},{'tick':1}],
                    error={'tick':1},guard_total_latency_s={'count':2})
        runner=SimpleNamespace(run=lambda **kwargs:{'cases':[record]})
        raw,report=wrapper.execute_wrapped(runner,module.ControlWorkerGC,case=0,steps=3,
                                         raw_output=Path('/tmp/nonexistent-reservation-wrapper-output'))
        self.assertEqual(report['attempted_control_steps'],2)
        self.assertEqual(report['completed_physical_words'],2)

    def test_modes_ids_counts_and_stability(self):
        smoke=wrapper.status([row(timing=False)],mode='smoke',expected_cases=1,steps=3,stable=True)
        self.assertTrue(smoke['requested_check_passed']);self.assertFalse(smoke['conditional_SIL_passed'])
        strict=wrapper.status([row(timing=False)],mode='validate',expected_cases=1,steps=3,stable=True)
        self.assertFalse(strict['requested_check_passed'])
        for records,stable in (([],True),([row(),row()],True),([row(case=1)],True),([row()],False)):
            self.assertFalse(wrapper.status(records,mode='smoke',expected_cases=1,steps=3,stable=stable)['requested_check_passed'])

    def test_parent_aggregation_mock_children_preserves_every_case(self):
        calls=[]
        def run(command,**kwargs):
            calls.append(command);case=int(command[command.index('--case')+1]);steps=int(command[command.index('--steps')+1])
            output=Path(command[command.index('--output')+1]);meta=Path(command[command.index('--metadata-output')+1])
            output.write_text(json.dumps({'cases':[row(case,steps)]}))
            meta.write_text(json.dumps({'source_hashes_stable':True,'source_sha256_before':{'f':'x'},
              'source_sha256_after':{'f':'x'},'gc':{'prior_enabled_state_restored':True,'collections_occurred_inside_control':False}}))
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as directory,patch.object(wrapper,'dependencies',return_value=(None,None)),\
             patch.object(wrapper,'snapshot',return_value={'f':'x'}),patch.object(wrapper.subprocess,'run',side_effect=run):
            output=Path(directory)/'aggregate.json'
            self.assertEqual(wrapper.parent(mode='smoke',cases=2,steps=3,output=output),0)
            report=json.loads(output.read_text());self.assertEqual([r['case'] for r in report['cases']],[0,1])
            self.assertTrue(report['all_requested_cases_passed']);self.assertEqual(len(calls),2)
            self.assertTrue(report['source_proof_input_hashes_stable'])
            self.assertNotIn('original_evidence_hashes_stable',report)
            self.assertTrue(all(c[c.index('--case')+1]==str(i) for i,c in enumerate(calls)))
            with self.assertRaises(FileExistsError):wrapper.parent(mode='smoke',cases=2,steps=3,output=output)
            with self.assertRaises(ValueError):wrapper.parent(mode='validate',cases=1,steps=3,output=Path(directory)/'other')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WrapperTests))
    (HERE/'nonflight_test_audit.json').write_text(json.dumps({'tests_run':result.testsRun,'passed':result.wasSuccessful(),
       'failures':len(result.failures),'errors':len(result.errors),'flight_executed':False,'timing_benchmark_executed':False,
       'flight_certificate':False},indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
