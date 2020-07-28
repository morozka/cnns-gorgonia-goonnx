package main

import (
	"fmt"

	"github.com/cnns-gorgonia-goonnx/DTPrediction/sqlreg"
	"github.com/cnns-gorgonia-goonnx/src/nntypes"
	grg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	dbData := sqlreg.LoadFullModel()
	//Config of network
	lps := make([]nntypes.Params, 0)
	lps = append(lps, nntypes.FulconParams{
		OutSize:      128,
		Activation:   "tanh",
		WeightsShape: []int{8, 128},
		Biased:       true,
	})
	lps = append(lps, nntypes.LSTMParams{
		OutputSize: 128,
	})

	lps = append(lps, nntypes.FulconParams{
		OutSize:      1,
		Activation:   "off",
		WeightsShape: []int{128, 1},
		Biased:       true,
	})

	wnet := nntypes.GenerateNet(lps, tensor.Shape{1, 8}, tensor.Shape{1})
	//train
	for i := 0; i < 10; i++ { //epochs
		for j := 0; j < len(dbData); j++ { //iters
			wnet.Machine.Reset()
			//normalization of train data
			input := []float64{dbData[j].Humidity / 100.0, (dbData[j].Pressure - 978) / 50.0, (dbData[j].Temp + 45) / 90.0, dbData[j].Visibility / 10000.0, BofortNormScale(dbData[j].WindSpeed), dbData[j].DayOfWeek / 6.0, dbData[j].Hour / 23.0, dbData[j].Month / 12.0}
			prb := dbData[j].DtpSum

			if prb > 2 {
				prb = 1
			} else {
				if prb == 0 {
					prb = -1
				} else {
					prb = 0
				}
			}
			output := []float64{prb}

			iten := tensor.New(tensor.WithShape(1, 8), tensor.WithBacking(input))
			oten := tensor.New(tensor.WithShape(1), tensor.WithBacking(output))
			grg.Let(wnet.Input, iten)
			grg.Let(wnet.LearnNode, oten)
			err := wnet.Machine.RunAll()
			if err != nil {
				fmt.Println(err)
			}
			err = wnet.Solver.Step(grg.NodesToValueGrads(wnet.Weights))
			if err != nil {
				fmt.Println(err)
			}

		}
	}

	ci := 0.0 //numder of missed risky hours
	cn := 0.0 //number of missed zeroes probs
	cc := 0.0 //number of zeroes in test

	cost := 0.0     //sum of MSE on tests
	excluded := 0.0 //number of low prob hours, that skiped during test
	//testing
	for j := 1000; j < 2000; j++ {
		wnet.Machine.Reset()

		input := []float64{dbData[j].Humidity / 100.0, (dbData[j].Pressure - 978) / 50.0, (dbData[j].Temp + 45) / 90.0, dbData[j].Visibility / 10000.0, BofortNormScale(dbData[j].WindSpeed), dbData[j].DayOfWeek / 6.0, dbData[j].Hour / 23.0, dbData[j].Month / 12.0}
		fmt.Println("Weather+Date Info:")
		fmt.Println(input)
		prb := dbData[j].DtpSum
		if prb > 2 {
			prb = 1
		} else {
			if prb == 0 {
				prb = -1
			} else {
				prb = 0
			}
		}
		output := []float64{prb}

		iten := tensor.New(tensor.WithShape(1, 8), tensor.WithBacking(input))
		oten := tensor.New(tensor.WithShape(1), tensor.WithBacking(output))
		grg.Let(wnet.Input, iten)
		grg.Let(wnet.LearnNode, oten)
		wnet.Machine.RunAll()
		fmt.Println("Expected  : ", output)
		fmt.Println("Prediction: ", wnet.TrueNode.Value().Data())
		if prb > 0.0 && wnet.TrueNode.Value().Data().(float64) < 0.0 {
			ci++
		}
		if prb < 0.0 && wnet.TrueNode.Value().Data().(float64) > 0.0 {
			cn++
		}
		if prb < 0.0 {
			cc++
		}
		if prb == 0.0 {
			excluded++
		}
		cost = cost + wnet.Cost.Value().Data().(float64)

	}
	fmt.Println(1000, len(dbData), ci, cn, cc, excluded, cost)
	wnet.Machine.Close()
}

//BofortNormScale - returns normalized value of wind speed by Bofort's scale
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
