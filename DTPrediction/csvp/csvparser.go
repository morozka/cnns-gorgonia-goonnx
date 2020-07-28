package csvp

/*


USE MAINSQL INSTEAD


*/

import (
	"bufio"
	"encoding/csv"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"time"
)

//indexes in csv file
const (
	temp       = 0
	pressure   = 1
	humidity   = 2
	visibility = 3
	wind_speed = 4
	clouds     = 5
	ts         = 6
	rts        = 7
	hour       = 8
	day        = 9
	month      = 10
	wday       = 11
)

//Weather - сведения о погоде
type Weather struct {
	Temp       float64
	Presssure  float64
	Humidity   float64
	Visibility float64
	WindSpeed  float64
	Clouds     float64
}

//DTS - структура для хранения информации о времени  дтп
type DTS struct {
	DayOfWeek float64
	Day       float64
	Month     float64
	Hour      float64
}

//APH - структура для дтп произошедших в один час
type APH struct {
	DateTime   string
	Conditions Weather
	Places     []string
}

//APHO - структура, которая должна быть оптимизированна для подачи на первый каскад нейросети
type APHO struct {
	Wth Weather
	Prb float64
	Dts DTS
}

//GetDatasets - возвращает два набора данных из файла с разделением по дате train, test
func GetDatasets(file string) ([]APHO, []APHO) /*(map[string]APHO, map[string]APHO)*/ {

	//train := make(map[string]APHO, 0)
	//test := make(map[string]APHO, 0)
	rawData := make([][]string, 0)
	train := make([]APHO, 0)
	test := make([]APHO, 0)
	csvFile, _ := os.Open(file)
	reader := csv.NewReader(bufio.NewReader(csvFile))
	for {
		line, error := reader.Read()
		if error == io.EOF {
			break
		} else if error != nil {
			log.Fatal(error)
		}
		rawData = append(rawData, line)
	}

	for i := 0; i < len(rawData); i = i + 1 {
		pl := ParseLine(rawData[i])

		if i <= 500 || i > 1000 {
			train = append(train, pl)
		} else {
			train = append(train, pl)
			test = append(test, pl)
		}
	}
	return train, test
}
func parse(s string) float64 {
	buf, _ := strconv.ParseFloat(s, 64)
	return float64(buf)
}
func CompareAPHO(a, b APHO) bool {
	if a.Wth.Clouds == b.Wth.Clouds {
		if a.Wth.Humidity == b.Wth.Humidity {
			if a.Wth.Presssure == b.Wth.Presssure {
				if a.Wth.Temp == b.Wth.Temp {
					if a.Wth.Visibility == b.Wth.Visibility {
						if a.Wth.WindSpeed == b.Wth.WindSpeed {
							return true
						}
					}
				}
			}
		}
	}
	return false
}
func Max(a, b float64) float64 {
	if a > b {
		return a
	} else {
		return b
	}
}
func dist(a, b float64) float64 {
	return math.Abs(a-b) / Max(a, b)
}
func ClasterAPHO(a, b APHO) bool {
	if dist(a.Wth.Clouds, b.Wth.Clouds) < 0.2 {
		if dist(a.Wth.Humidity, b.Wth.Humidity) < 0.2 {
			if dist(a.Wth.Presssure, b.Wth.Presssure) < 0.2 {
				if dist(a.Wth.Temp, b.Wth.Temp) < 0.2 {
					if dist(a.Wth.Visibility, b.Wth.Visibility) < 0.2 {
						if dist(a.Wth.WindSpeed, b.Wth.WindSpeed) < 0.2 {
							if dist(a.Dts.DayOfWeek, b.Dts.DayOfWeek) == 0 {
								if dist(a.Dts.Hour, b.Dts.Hour) < 0.2 {
									return true
								}
							}
						}
					}
				}
			}
		}
	}
	return false
}
func ParseLine(line []string) APHO {
	w := Weather{
		Temp:       (parse(line[temp]) + 45) / 90.0,
		Presssure:  (parse(line[pressure]) - 978) / 50.0,
		Humidity:   parse(line[humidity]) / 100.0,
		Visibility: parse(line[visibility]) / 10000.0,
		WindSpeed:  BofortNormScale(parse(line[wind_speed])),
		Clouds:     parse(line[clouds]) / 100.0,
	}

	prb := parse(line[rts]) / 10.0

	if prb > 0.2 {
		prb = 1.0
	} else {
		if prb == 0 {
			prb = -1.0
		} else {
			prb = 0.0
		}
	}
	return APHO{
		Wth: w,
		Dts: ParseDate(line[ts]),
		Prb: prb,
	}
}
func BofortNormScale(sp float64) float64 {
	switch {
	case sp < 0.6:
		return 0.0
	case sp < 1.7:
		return 1.0 / 12.0
	case sp < 3.4:
		return 2.0 / 12.0
	case sp < 5.3:
		return 3.0 / 12.0
	case sp < 7.5:
		return 4.0 / 12.0
	case sp < 9.9:
		return 5.0 / 12.0
	case sp < 12.5:
		return 6.0 / 12.0
	case sp < 15.3:
		return 7.0 / 12.0
	case sp < 18.3:
		return 8.0 / 12.0
	case sp < 21.6:
		return 9.0 / 12.0
	case sp < 25.2:
		return 10.0 / 12.0
	case sp < 29:
		return 11.0 / 12.0
	default:
		return 1.0
	}
}
func AVGAPHO(a, b APHO) APHO {
	var ans APHO
	ans.Dts.Day = (a.Dts.Day + b.Dts.Day) / 2
	ans.Dts.Hour = (a.Dts.Hour + b.Dts.Hour) / 2
	ans.Dts.Month = (a.Dts.Month + b.Dts.Month) / 2
	ans.Dts.DayOfWeek = (a.Dts.DayOfWeek + b.Dts.DayOfWeek) / 2
	ans.Wth.Clouds = (a.Wth.Clouds + b.Wth.Clouds) / 2
	ans.Wth.Humidity = (a.Wth.Humidity + b.Wth.Humidity) / 2
	ans.Wth.Visibility = (a.Wth.Visibility + b.Wth.Visibility) / 2
	ans.Wth.Presssure = (a.Wth.Presssure + b.Wth.Presssure) / 2
	ans.Wth.WindSpeed = (a.Wth.WindSpeed + b.Wth.WindSpeed) / 2
	ans.Wth.Temp = (a.Wth.Temp + b.Wth.Temp) / 2
	ans.Prb = (a.Prb + b.Prb) / 2
	return ans
}
func SumAPHO(a, b APHO) APHO {
	var ans APHO
	ans.Dts.Day = (a.Dts.Day + b.Dts.Day)
	ans.Dts.Hour = (a.Dts.Hour + b.Dts.Hour)
	ans.Dts.Month = (a.Dts.Month + b.Dts.Month)
	ans.Dts.DayOfWeek = (a.Dts.DayOfWeek + b.Dts.DayOfWeek)
	ans.Wth.Clouds = (a.Wth.Clouds + b.Wth.Clouds)
	ans.Wth.Humidity = (a.Wth.Humidity + b.Wth.Humidity)
	ans.Wth.Visibility = (a.Wth.Visibility + b.Wth.Visibility)
	ans.Wth.Presssure = (a.Wth.Presssure + b.Wth.Presssure)
	ans.Wth.WindSpeed = (a.Wth.WindSpeed + b.Wth.WindSpeed)
	ans.Wth.Temp = (a.Wth.Temp + b.Wth.Temp)
	ans.Prb = (a.Prb + b.Prb)
	return ans
}
func DivAPHO(a APHO, i float64) APHO {
	var ans APHO
	ans.Dts.Day = a.Dts.Day / i
	ans.Dts.Hour = a.Dts.Hour / i
	ans.Dts.Month = a.Dts.Month / i
	ans.Dts.DayOfWeek = a.Dts.DayOfWeek / i
	ans.Wth.Clouds = a.Wth.Clouds / i
	ans.Wth.Humidity = a.Wth.Humidity / i
	ans.Wth.Visibility = a.Wth.Visibility / i
	ans.Wth.Presssure = a.Wth.Presssure / i
	ans.Wth.WindSpeed = a.Wth.WindSpeed / i
	ans.Wth.Temp = a.Wth.Temp / i
	ans.Prb = a.Prb / i
	return ans
}
func ParseDate(line string) DTS {
	rts, _ := time.Parse("2006-01-02 15:04:05", line)
	var d DTS
	d.Day = float64(rts.Day()) / 31.0
	d.Month = float64(rts.Month()) / 12.0
	d.Hour = float64(rts.Hour()) / 23.0
	d.DayOfWeek = float64(rts.Weekday()) / 6.0
	return d
}
