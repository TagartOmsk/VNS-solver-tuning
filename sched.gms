Option OptCA = 0;
Option OptCR = 0;
Option mip = gurobi;
Sets
$include %sets%

alias (n, n_alias);
alias (m, m_alias);

Scalars
         Tmax max duration of one mode;


Parameters
         s(p) quantity of processor tacts till completing job p

         v(p, m) 'execution speed job p in mode m'

         q(p, m) 'job p is in progress in mode m'

         a(m, m_alias) 'm is started after m_alias';

a("0","0")=0;

$include %instance%

Variables
         t(n, m) quantity of processor tacts which is needed for mode m in event point n
         Cmax max time;

Binary variable d(n, m) mode m is being executed in event point n;
Binary variable y(p, n) auxiliary variable which is meaning increment estimate;

Equations
         time_sum sum of time
         non_negative_mode_duration(n, m) duration of mode m in event point n is non-negative
         zero_mode_duration(n, m) duration of mode m is 0 if the mode isnt executed in event point n
         mode_quantity(n) quantity of modes in each event point is 1
         event_points_quantity(m) quantity of event points for each mode is 1
         full_job(p) each job is fully completed
         increment_estimate(p) auxiliary equation for the next equation
         continuity(p, n) continuity of job execution
         partial_order(m, m_alias) partial order of mode execution
         zero_event_point at the zero event point zero mode is executed4;

         time_sum.. sum(n, sum(m, t(n, m))) =l= Cmax;
         non_negative_mode_duration(n, m).. t(n, m) =g= 0;
         zero_mode_duration(n, m).. t(n, m) =l= d(n, m) * Tmax;
         mode_quantity(n).. sum(m, d(n, m)) =l= 1;
         event_points_quantity(m).. sum(n, d(n, m)) =l= 1;
         full_job(p).. sum(n, sum(m, t(n, m) * v(p, m))) =e= s(p);
         increment_estimate(p).. sum(n, y(p, n)) =e= 1;
         continuity(p, n).. sum(m, d(n, m) * q(p, m)) - sum(m, d(n - 1, m) * q(p, m)) =l= y(p, n);
         partial_order(m, m_alias)$(a(m, m_alias)>0).. sum(n, ord(n) * d(n, m_alias)) + 1 =l= (1 - sum(n, d(n, m))) * 35 + sum(n, ord(n) * d(n, m));
         zero_event_point.. d('1', '0') =e= 1;

Model mymodel /all/;
solve mymodel using mip minimizing Cmax;
display t.l, d.l;

File results / results_.txt /;
put results /;
put "estimated: " mymodel.objest /;
put "actual: " mymodel.objval /;
put "absolute gap: " abs(mymodel.objval-mymodel.objest) /;
put "solution: "Cmax.l /;
put "k           c                   t" /;
loop((n,m)$(t.l(n,m)>0),
     put n.tl, m.tl, t.l(n,m) /
) ;
putclose;
