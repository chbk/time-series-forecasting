--%% Create tables -------------------------------------------------------------

create table if not exists station (
  snapshot_id integer,
  station_id integer,
  snapshot_date text,
  name text,
  capacity integer,
  latitude real,
  longitude real,
  proxy_id integer,
  proxy_latitude real,
  proxy_longitude real,
  proxy_distance real,
  primary key (snapshot_id)
);

create table if not exists trip (
  trip_id integer,
  start_datetime text,
  stop_datetime text,
  start_station_id integer,
  start_station_name text,
  stop_station_id integer,
  stop_station_name text,
  primary key (trip_id),
  foreign key (start_station_id, start_station_name)
    references station (station_id, name),
  foreign key (stop_station_id, stop_station_name)
    references station (station_id, name)
);

--%% Populate tables -----------------------------------------------------------

.mode csv
.import '| tail -n +2 data/divvy_stations_2013-2017.csv' station
.import '| tail -n +2 data/divvy_trips_2013-2017.csv' trip

select * from station limit 10;
select * from trip limit 10;

--%% Add columns for station snapshot ids --------------------------------------

alter table trip
add column start_station_snapshot_id integer references station (snapshot_id);

alter table trip
add column stop_station_snapshot_id integer references station (snapshot_id);

--%% Join trip and station on station id, name, closest snapshot date ----------

update trip set
start_station_snapshot_id = coalesce(
  (
    select s_.snapshot_id
    from station as s_ left join trip as t_
    on s_.station_id = t_.start_station_id
    where t_.trip_id = trip.trip_id
    and s_.station_id = trip.start_station_id
    and s_.name = trip.start_station_name
    order by abs(
      strftime('%s', t_.start_datetime) - strftime('%s', s_.snapshot_date)
    )
    limit 1
  ),
  (
    select s_.snapshot_id
    from station as s_ left join trip as t_
    on s_.station_id = t_.start_station_id
    where t_.trip_id = trip.trip_id
    and s_.station_id = trip.start_station_id
    order by abs(
      strftime('%s', t_.start_datetime) - strftime('%s', s_.snapshot_date)
    )
    limit 1
  )
),
stop_station_snapshot_id = coalesce(
  (
    select s_.snapshot_id
    from station as s_ left join trip as t_
    on s_.station_id = t_.stop_station_id
    where t_.trip_id = trip.trip_id
    and s_.station_id = trip.stop_station_id
    and s_.name = trip.stop_station_name
    order by abs(
      strftime('%s', t_.stop_datetime) - strftime('%s', s_.snapshot_date)
    )
    limit 1
  ),
  (
    select s_.snapshot_id
    from station as s_ left join trip as t_
    on s_.station_id = t_.stop_station_id
    where t_.trip_id = trip.trip_id
    and s_.station_id = trip.stop_station_id
    order by abs(
      strftime('%s', t_.stop_datetime) - strftime('%s', s_.snapshot_date)
    )
    limit 1
  )
);

--%% Output trips --------------------------------------------------------------

.headers on
.mode list
.separator ,
.once 'data/divvy_trips_2013-2017.csv'

select
  trip_id,
  start_datetime,
  stop_datetime,
  start_station_snapshot_id,
  stop_station_snapshot_id
from trip;
