package sqlreg

import (
	"log"

	"github.com/go-pg/pg"
)

type FullModel struct {
	// TableName  struct{} `sql:"weather.weather" json:"-"`
	Temp       float64   `sql:"temp"`
	Pressure   float64   `sql:"pressure"`
	Humidity   float64   `sql:"humidity"`
	Visibility float64   `sql:"visibility"`
	WindSpeed  float64   `sql:"wind_speed"`
	Clouds     float64   `sql:"clouds"`
	Ts         string    `sql:"ts"`
	DtpSum     float64   `sql:"dtpsum"`
	Hour       float64   `sql:"hour"`
	DayOfWeek  float64   `sql:"dow"`
	Month      float64   `sql:"month"`
	DtpByPlace []float64 `pg:"dtpbyplace,array"` //`sql:"dtpbyplace"`
}

type AGGDATA struct {
	Name        string  `sql:"place"`
	Temperature float64 `sql:"temp"`
	Pressure    float64 `sql:"pressure"`
	Humidity    float64 `sql:"humidity"`
	Visibility  float64 `sql:"visibility"`
	WindSpeed   float64 `sql:"wind_speed"`
	Clouds      float64 `sql:"clouds"`
	DTP         float64 `sql:"dtp"`
}

func LoadFullModel() []FullModel {
	sqlString := `
	with place_weather_percent as (
		with places as (
			  select
				name as place,
			temp,
			  pressure,
			  humidity,
			  visibility,
			  wind_speed,
			  clouds,
	  to_timestamp(to_char("current", 'YYYY-MM-DD HH24:00:00'), 'YYYY-MM-DD HH24:00:00') as ts
			  from road_accidents.narrow_places_new_angelina as narrow_places
			  cross join weather.weather
			  where "current" between '2018-08-20 00:00:00' and '2019-08-20 00:00:00'
			  and narrow_places.radius_geom is not null and narrow_places.deleted <> true
			  order by place, ts asc
		)
		select
		  *
		from places
	  ), agg_dtp as (
		select
		  --dtp.uuid as dtp_uuid,
		  to_timestamp(to_char(dtp.ts, 'YYYY-MM-DD HH24:00:00'), 'YYYY-MM-DD HH24:00:00') as ts,
		  places.name as dtp_place
		from road_accidents.yandex_moscow as dtp
		join road_accidents.narrow_places_new_angelina as places
		on st_contains(places.radius_geom, dtp.geom)
		where dtp.ts between '2018-08-20 00:00:00' and '2019-08-20 00:00:00' and places.deleted <> true
		group by dtp_place, ts
		order by ts asc
	  ),pend as(
	  select cdtp.place,place_weather_percent.temp,place_weather_percent.pressure,
			  place_weather_percent.humidity,
			place_weather_percent.visibility,
			 place_weather_percent.wind_speed,
			place_weather_percent.clouds, place_weather_percent.ts as ts, 
			  case when lead(cdtp.place)over(order by cdtp.place,place_weather_percent.ts desc)=cdtp.place and 
		  lead(agg_dtp.ts)over(order by cdtp.place,place_weather_percent.ts desc) is not null 
			  then null else agg_dtp.ts end as realdtp,
			  extract(hour from place_weather_percent.ts) as hour,extract(DOW from place_weather_percent.ts) as dow,
			  extract(month from place_weather_percent.ts) as month
			  from 
	  (select 
	  count(agg_dtp.dtp_place) as numdtp, place,count(place) as numts
	  from place_weather_percent
	  left join agg_dtp
	  on place_weather_percent.ts = agg_dtp.ts
	  and place_weather_percent.place = agg_dtp.dtp_place 
	  group by place
	  order by count(agg_dtp.dtp_place) desc) as cdtp
	  left join place_weather_percent on
	  place_weather_percent.place=cdtp.place
	  left join agg_dtp
	  on place_weather_percent.ts = agg_dtp.ts
	  and place_weather_percent.place = agg_dtp.dtp_place 
	  group by cdtp.numdtp,cdtp.place, place_weather_percent.temp,place_weather_percent.pressure,
			  place_weather_percent.humidity,
			  place_weather_percent.visibility,
			  place_weather_percent.wind_speed,
			  place_weather_percent.clouds, place_weather_percent.ts,agg_dtp.ts 
	  order by cdtp.place,place_weather_percent.ts asc
	  ), gpf as(
	  select pend.place, count(pend.realdtp) as nrdtp from pend group by pend.place
	  )
	  
	  select pend.temp,pend.pressure,
			  pend.humidity,
			  pend.visibility,
			 pend.wind_speed,
			 pend.clouds, pend.ts,count(pend.realdtp) as dtpsum,pend.hour,pend.dow, pend.month,
			 array_agg(case when pend.realdtp is not null then 1 else 0 end order by pend.place) as dtpbyplace
			 --count(pend.realdtp)
	  from pend
	  right join gpf on pend.place=gpf.place
	  where gpf.nrdtp>150
	  group by pend.temp,pend.pressure,
			  pend.humidity,
			  pend.visibility,
			 pend.wind_speed,
			 pend.clouds, pend.ts,pend.ts,pend.hour,pend.dow,pend.month
	  order by pend.ts asc;
	 `

	//sql close comment
	var err error
	dbConn := pg.Connect(&pg.Options{
		Addr:     "172.20.0.73:5432",
		User:     "s_morozov",
		Password: "s_morozov",
		Database: "nii",
	})
	defer dbConn.Close()
	var dbData []FullModel
	_, err = dbConn.Query(&dbData, sqlString)
	if err != nil {
		log.Panicln(err)
	}
	return dbData
}
