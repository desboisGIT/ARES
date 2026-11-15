from pathlib import Path
import sys,unittest,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation_common import verdict,operational_hashes
class VerdictTests(unittest.TestCase):
    def good(self):return {'completed_steps':2,'positive_certificates':2,'actual_C_misses':0,'collisions':0,'unknown_entries':0,'queue_mismatches':0,'sources_stable':True,'operational_sources_stable':True,'delivery_modules_imported':[],'control_worker_gc':{'prior_enabled_state_restored':True,'collections_occurred_inside_control':False},'error':None,'all_in_shield_control_above10ms':0,'all_in_shield_control_max_s':.009,'mission_success':True}
    def test_strict_mission_and_wall_timing(self):
        x=self.good();self.assertTrue(verdict(x)['requested_checks_passed'])
        for patch in ({'mission_success':False},{'all_in_shield_control_above10ms':1},{'all_in_shield_control_max_s':.010001},{'delivery_modules_imported':['ares.verified_safety']},{'all_in_shield_control_max_s':None},{'error':{'type':'Setup'}}):
            bad={**x,**patch};self.assertFalse(verdict(bad)['requested_checks_passed'])
    def test_missing_none_and_parse_fallback_reports_fail_without_raising(self):
        for bad in [{},{'completed_steps':None},{'completed_steps':None,'error':{'type':'WorkerReportParseError'}},{'completed_steps':True,'positive_certificates':True}]:
            self.assertFalse(verdict(bad)['requested_checks_passed'])
    def test_only_three_unused_delivery_files_excluded(self):
        x={'ares/flight_env.py':'a','ares/safety_contract.py':'b','ares/verified_safety/_launcher.py':'c','reports/supplier.py':'d'}
        self.assertEqual(set(operational_hashes(x)),{'ares/flight_env.py','ares/safety_contract.py','reports/supplier.py'})
if __name__=='__main__':unittest.main(verbosity=2)
