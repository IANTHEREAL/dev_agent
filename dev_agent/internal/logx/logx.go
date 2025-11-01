package logx

import (
    "fmt"
    "os"
    "time"
)

type Level int

const (
    Debug Level = iota
    Info
    Warning
    Error
)

var current Level = Info
var loggerName = "dev_agent"

func SetLevel(l Level) { current = l }

func ts() string { return time.Now().Format("15:04:05") }

func Infof(format string, args ...any) {
    if current <= Info {
        fmt.Fprintf(os.Stdout, "[%s] INFO %s: ", ts(), loggerName)
        fmt.Fprintf(os.Stdout, format+"\n", args...)
    }
}

func Warningf(format string, args ...any) {
    if current <= Warning {
        fmt.Fprintf(os.Stdout, "[%s] WARNING %s: ", ts(), loggerName)
        fmt.Fprintf(os.Stdout, format+"\n", args...)
    }
}

func Errorf(format string, args ...any) {
    if current <= Error {
        fmt.Fprintf(os.Stderr, "[%s] ERROR %s: ", ts(), loggerName)
        fmt.Fprintf(os.Stderr, format+"\n", args...)
    }
}

func Debugf(format string, args ...any) {
    if current <= Debug {
        fmt.Fprintf(os.Stdout, "[%s] DEBUG %s: ", ts(), loggerName)
        fmt.Fprintf(os.Stdout, format+"\n", args...)
    }
}

