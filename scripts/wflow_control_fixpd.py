"""Generate init text files for CONTROL maizsim simulations."""

from ideotype.log import log_fetchinfo
from ideotype.wflow_setup import (make_dircts, make_inits, make_cultivars,
                                  make_runs, make_jobs, make_subjobs)

log_fetchinfo('control_fixpd')
make_dircts('control_fixpd', cont_cvars=False)
make_inits('control_fixpd', cont_cvars=False)
make_cultivars('control_fixpd', cont_cvars=False)
make_runs('control_fixpd', cont_cvars=False)
make_jobs('control_fixpd', cont_cvars=False)
make_subjobs('control_fixpd')
