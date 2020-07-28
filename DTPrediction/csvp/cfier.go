package csvp

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
)

func CalcDists(file string) {
	rawData := make([][]string, 0)
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
	adrs := make([]string, 0)
	dtps := make([][]int, 0)
	dtps = append(dtps, make([]int, 0))
	dtps[0] = append(dtps[0], checkDtp(rawData[1]))
	adrs = append(adrs, rawData[1][0])
	j := 0
	for i := 2; i < len(rawData); i++ {
		if !checkPlace(rawData[i], rawData[i-1]) {
			j++
			adrs = append(adrs, rawData[i][0])
			dtps = append(dtps, make([]int, 0))
		}
		dtps[j] = append(dtps[j], checkDtp(rawData[i]))
	}
	answ := make([][]int, 0)
	for i := 0; i < len(dtps); i++ {
		answ = append(answ, make([]int, 0))
		for z := 0; z < len(dtps); z++ {
			if i == z {
				answ[i] = append(answ[i], 0)
				continue
			}
			answ[i] = append(answ[i], 0)
			for k := 0; k < len(dtps[i]) && k < len(dtps[z]); k++ {

				if dtps[i][k] == dtps[z][k] && dtps[i][k] == 1 {
					answ[i][len(answ[i])-1]++
				}
			}
		}
	}
	for i := 0; i < len(answ); i++ {
		for k := 0; k < len(answ[i]); k++ {
			if answ[i][k] > 200 && answ[i][k] != 0 {
				fmt.Println(i, k, adrs[i], adrs[k], len(dtps[i]), len(dtps[k]), answ[i][k])
			}
		}
	}
}
func checkDtp(line []string) int {
	if line[8] != "" {
		return 1
	}
	return 0
}
func checkPlace(a []string, b []string) bool {
	if a[0] != b[0] {
		return false
	}
	return true
}
