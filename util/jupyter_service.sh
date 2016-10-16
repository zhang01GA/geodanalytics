#!/bin/bash

# Source function library.
# . /etc/init.d/functions

# Purpose:  jupyter control script
# Usage:    jupyter_ctrl.sh start stop status
#
# set environment vars for jupyter to use.

# uncomment the module load lines to enable notebooks to use them
# Alternateivly, dynamically modify your pythonpath in each notebook
#module use /g/data/v10/public/modules/modulefiles --append
#module load agdc-py2-prod
#module load agdc-py2-dev


# jupyter notebook --port=4999 &> jupyter_notebook.log  &
# jupyter notebook --no-browser   --port=4999 &> jupyter_notebook.log  &

# log file
LOG_FILE=tmpdir/jupyter_notebook.log
PROC_NAME='jupyter-notebook'
status(){ 
    echo Checking if jupyter-notebook is already running? 
    jpid=`pgrep -u $USER -f $PROC_NAME`
    echo The $PROC_NAME pid is: $jpid

    if [ ${jpid}1 -gt 1 ]; then
        echo "It ($jpid) is an integer, will return 1"
        return 1
        #return $jpid  # this number will be cast into [0 256) eg: 8625 -> 177
    else
        echo  It $jpid is empty or not defined, will return 0
        return 0
    fi
}


start() {
    
    echo first check if the service is started already
    echo "If not, do start"
    # code to start app comes here 
    # example: daemon program_name &

    echo `date` > $LOG_FILE

    echo Starting jupyter on $HOSTNAME >> $LOG_FILE 

    echo " --------------------------------------------------" >> $LOG_FILE

    jupyter notebook &>> $LOG_FILE &

    echo jupyer has been started: `pgrep jupyter`

}

stop() {
    # code to stop app comes here 
    # example: killproc program_name
    echo 'Stopping jupyter now..'
    echo 'Running  kill -9 `pgrep jupyter` '
    kill -9 `pgrep jupyter` 

    status
}

case "$1" in 
    start)
       status
       retval=$?

       if [ $retval -eq 0 ]; then
           start
       else
           echo  $PROC_NAME is already running
           echo $retval
       fi
       ;;
    stop)
       stop
       ;;
    restart)
       stop
       start
       ;;
    status)
       # code to check status of app comes here 
       # example: status program_name
       status
       retval=$?

       if [ $retval -eq 0 ]; then
          echo "Not running " 
       else
           echo  $PROC_NAME is already running
           echo $retval
       fi
       ;;
    *)
       echo "Usage: $0 {start|stop|status|restart}"
esac

exit 0


