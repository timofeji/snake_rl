#import "Basic";
#import "Random";
#import "Math";

WIDTH:  u32 = 30;
HEIGHT: u32 = 30;

#if OS == .WINDOWS {

    CONSOLE_FONT::"Consolas";
    FOREGROUND_INTENSITY       :: 0x0008;
    FOREGROUND_BLACK           :: 0x0000;
    FOREGROUND_BLUE            :: 0x0001;
    FOREGROUND_GREEN           :: 0x0002;
    FOREGROUND_RED             :: 0x0004;

    BACKGROUND_BLACK           :: 0x0070;
    BACKGROUND_BLUE            :: 0x0010;
    BACKGROUND_GREEN           :: 0x0020;
    BACKGROUND_RED             :: 0x0040;



    ATTACH_PARENT_PROCESS :: 0xFFFFFFFF; 

    GENERIC_WRITE  :: 0x40000000;
    GENERIC_READ   :: 0x80000000;
    CONSOLE_TEXTMODE_BUFFER :: 0x0001;
    FILE_SHARE_READ    :: 0x00000001;
    FILE_SHARE_WRITE   :: 0x00000002;
    FONT_SIZE :: 32;

    CONSOLE_ORIGIN : COORD ;

    buffer: [..] CHAR_INFO;
    currentBuffer : u32;
    hBuffer : [2] HANDLE;


    cfi :CONSOLE_FONT_INFOEX;

    init_console ::() {
        hBuffer[0] = CreateConsoleScreenBuffer(GENERIC_READ | GENERIC_WRITE ,FILE_SHARE_READ | FILE_SHARE_WRITE, null, CONSOLE_TEXTMODE_BUFFER, null);
        hBuffer[1] = CreateConsoleScreenBuffer(GENERIC_READ | GENERIC_WRITE ,FILE_SHARE_READ | FILE_SHARE_WRITE, null, CONSOLE_TEXTMODE_BUFFER, null);

        hConsole: HANDLE  = GetStdHandle(STD_OUTPUT_HANDLE);

        

        if (GetCurrentConsoleFontEx(hConsole, true, *cfi)) {
            // Set the size to be square (e.g., 16x16 pixels)
            cfi.dwFontSize.x = 6*FONT_SIZE;
            cfi.dwFontSize.y = FONT_SIZE;
            memcpy(*cfi.FaceName, CONSOLE_FONT.data, CONSOLE_FONT.count);


            // Use a good monospaced font like Consolas

            // SetCurrentConsoleFontEx(hBuffer[0], true, *cfi);
            // SetCurrentConsoleFontEx(hBuffer[1], true, *cfi);
        } else {
            print("Warning: Could not get console font info. Proceeding with default font.\n");
        }



    }

    // Restore the original default buffer
    close_game ::() {
        CloseHandle(hBuffer[0]);
        CloseHandle(hBuffer[1]);
        SetConsoleActiveScreenBuffer(GetStdHandle(STD_OUTPUT_HANDLE));
    }

    flip_buffers ::() {
        SetConsoleActiveScreenBuffer(hBuffer[currentBuffer]);
        currentBuffer = 1 - currentBuffer;
    }

    draw_game :: (){

       grid := game.state;
   
        hCurrent : HANDLE = hBuffer[currentBuffer];
        cursorInfo : CONSOLE_CURSOR_INFO;
        GetConsoleCursorInfo(hCurrent, *cursorInfo);

        originalVisibility := cursorInfo.bVisible;
        cursorInfo.bVisible = false;
        SetConsoleCursorInfo(hCurrent, *cursorInfo);


        // get current console info
        consoleInfo : CONSOLE_SCREEN_BUFFER_INFO;
        GetConsoleScreenBufferInfo(hCurrent, *consoleInfo);


        SetCurrentConsoleFontEx(hCurrent, true, *cfi);

        // array_reset(*fontName);
        // array_reserve(fontName, CONSOLE_FONT.count );


    

        
        // cellCount := cast(u32) (consoleInfo.dwSize.x * consoleInfo.dwSize.y);
        
        screen_x := cast(u32)consoleInfo.dwSize.x;
        screen_y := cast(u32)consoleInfo.dwSize.y;
        cellCount := cast(u32) (consoleInfo.dwSize.x * consoleInfo.dwSize.y);
        array_reset(*buffer);
        array_reserve(*buffer, cellCount);

   
        for i:0..cellCount - 1 {
            c := CHAR_INFO.{#char " ",  FOREGROUND_INTENSITY};
            c.Char =  cast(u16) (random_get() % (1 << 16));
            

            if(i % screen_x < WIDTH && i / screen_x < HEIGHT) {
               grid_index := (i % screen_x) + (i / screen_x) * WIDTH;
                if grid[grid_index] == SNAKE {
                    c.Char = #char " ";
                    c.Attributes = BACKGROUND_BLACK  ;
                } else if grid[grid_index] == FOOD{
                    c.Char = #char "@";
                    c.Attributes = BACKGROUND_RED ;
                }
                else {
                    c.Char = #char " ";
                    c.Attributes = FOREGROUND_GREEN | FOREGROUND_INTENSITY;
                }
            }




            if(i % screen_x < WIDTH && i / screen_x > HEIGHT - 1 && i / screen_x < HEIGHT+1) {
                c.Char = #char ".";
                c.Attributes = FOREGROUND_RED | FOREGROUND_INTENSITY;

                x :=  i % screen_x;
                if(x < 9){

                c.Char = cast(u16)(game.score/ pow10[8 - x]) % 10 + #char "0";
                }

            }
    
            array_add(*buffer, c);

        }


        // SetConsoleCursorPosition(hCurrent, CONSOLE_ORIGIN);
        WriteConsoleOutputA (hCurrent, buffer.data, consoleInfo.dwSize, CONSOLE_ORIGIN, *consoleInfo.srWindow);

        // cursorInfo.bVisible = originalVisibility;
        // SetConsoleCursorInfo(hConsole, *cursorInfo);
    }

    #scope_file;

    pow10_table::() -> string {
        builder: String_Builder;
        print_to_builder(*builder, "pow10 := u32.[");
        for i: 0..9-1 {
            if i > 0  print_to_builder(*builder, ", ");
            value: u32 = 1;
            for 0..i-1 value *= 10;
            print_to_builder(*builder, "%", value);
        }

        print_to_builder(*builder, "];");
        return builder_to_string(*builder);
    }
    #insert #run pow10_table();




    SetConsoleWindowInfo :: (hConsoleOutput: HANDLE, bAbsolute : bool, lpConsoleWindow : *SMALL_RECT) -> BOOL #foreign kernel32;
    SetConsoleScreenBufferSize :: (hConsoleOutput: HANDLE, dwSize : COORD) -> BOOL #foreign kernel32;
    SetConsoleActiveScreenBuffer :: (hConsoleOutput: HANDLE) -> BOOL #foreign kernel32;
    GetConsoleCursorInfo :: (hConsoleOutput: HANDLE, lpConsoleCursorInfo : *CONSOLE_CURSOR_INFO) -> BOOL #foreign kernel32;
    SetConsoleCursorInfo :: (hConsoleOutput: HANDLE, lpConsoleCursorInfo : *CONSOLE_CURSOR_INFO) -> BOOL #foreign kernel32;
    SetConsoleCursorPosition :: (hConsoleOutput: HANDLE, dwCursorPosition : COORD) -> BOOL #foreign kernel32;
    CreateConsoleScreenBuffer :: (dwDesiredAccess: u32, dwShareMode: u32, lpSecurityAttributes : *void, dwFlags : u32, lpScreenBufferData : *void) -> HANDLE #foreign kernel32;
    CloseHandle :: (hObject: HANDLE) -> BOOL #foreign kernel32;
    WriteConsoleA :: (hConsoleOutput: HANDLE, lpBuffer: *void, nNumberOfCharsToWrite : u32, lpNumberOfCharsWritten : *u32, lpReserved : *void) -> BOOL #foreign kernel32;
    WriteConsoleOutputA :: (hConsoleOutput: HANDLE, lpBuffer: *CHAR_INFO, dwBufferSize : COORD, dwBufferCoord : COORD, lpWriteRegion : *SMALL_RECT) -> BOOL #foreign kernel32;
    FillConsoleOutputAttribute :: (hConsoleOutput: HANDLE, wAttribute: u16, nLength : u32, dwWriteCoord : COORD, lpNumberOfAttrsWritten : *u32) -> BOOL #foreign kernel32;
    FillConsoleOutputCharacterA :: (hConsoleOutput: HANDLE, cCharacter: u16, nLength : u32, dwWriteCoord : COORD, lpNumberOfCharsWritten : *u32) -> BOOL #foreign kernel32;
    AttachConsole :: (dwProcessId: u32) -> BOOL #foreign kernel32;
    AllocConsole :: () -> BOOL #foreign kernel32;
    GetConsoleScreenBufferInfo :: (hConsoleOutput: HANDLE, lpConsoleScreenBufferInfo : *CONSOLE_SCREEN_BUFFER_INFO) -> BOOL #foreign kernel32;
    SetConsoleTextAttribute :: (hConsoleOutput: HANDLE, wAttributes : u16) -> BOOL #foreign kernel32;
    GetStdHandle :: (handle: s32) -> HANDLE #foreign kernel32;
    Sleep :: (dwMilliseconds: u32) -> void #foreign kernel32;



    GetCurrentConsoleFontEx :: (hConsoleOutput: HANDLE,  bMaximumWindow:bool, cfi:*CONSOLE_FONT_INFOEX) -> BOOL #foreign kernel32;
    SetCurrentConsoleFontEx :: (hConsoleOutput: HANDLE,  bMaximumWindow:bool, cfi:*CONSOLE_FONT_INFOEX) -> BOOL #foreign kernel32;

    HANDLE :: *void;



    CONSOLE_FONT_INFOEX  ::  struct {
        cbSize: u64 ;
        nFont: u32 ;
        dwFontSize: COORD ;
        FontFamily: u32  ;
        FontWeight: u32  ;
        FaceName:[] u16 ;
    }

    CHAR_INFO :: struct {
        Char: u16;
        Attributes: u16;
    }
    
    SMALL_RECT :: struct {
        left, top, right, bottom: s16;
    }
    
    COORD :: struct {
        x, y: s16;
    }
    CONSOLE_CURSOR_INFO :: struct {
        dwSize: u32;
        bVisible: bool;
    }

    CONSOLE_SCREEN_BUFFER_INFO :: struct {
        dwSize, dwCursorPosition: COORD;
        wAttributes: u16;
        srWindow: SMALL_RECT;
        dwMaximumWindowSize: COORD;
    }

    BOOL     :: enum s32 { FALSE :: 0; TRUE :: 1; }
    
    STD_OUTPUT_HANDLE :: -11;
    
    kernel32 :: #foreign_system_library "kernel32";
} else {
    FOREGROUND_INTENSITY       :: 1;
    FOREGROUND_BLUE            :: 34;
    FOREGROUND_GREEN           :: 32;
    FOREGROUND_RED             :: 31;
    BACKGROUND_BLUE            :: 44;
    BACKGROUND_GREEN           :: 42;
    BACKGROUND_RED             :: 41;

    // Linux version:

    // @Hack Ugly and temporary! Just want Linux terminal color for the test framework.
    //       We'll need a cross-platform solution for Windows, Linux, and Mac
    print_color :: (format_string : string, args : .. Any, color : u8) {
        builder : String_Builder;
        defer free_buffers(*builder);
        print_to_builder(*builder, format_string, ..args);

        s := builder_to_string(*builder);
        defer free(s);

        print("\e[0;%m%", color, s);
        // Reset console
        print("\e[0;m");
    }

}





Game :: struct {
    state : [#run WIDTH * HEIGHT] u32;
    flow  : [#run WIDTH * HEIGHT] u32;


    head: u32;  
    tail: u32;
    food: u32;
    
    snake_ptr: *u32;
    snake_length: u32 = 5;

    score:  u16;

    direction: s8 = 1;        // 0=up, 1=right, 2=down, 3=left
    is_game_over: bool;
}

game: Game;

// States that the game can be in
EMPTY:: 0;
SNAKE :: 1;
FOOD:: 2;
CORPSE:: 3;
WALL:: 4;

place_food ::()
{
    food_x := random_get() % WIDTH;
    food_y := random_get() % HEIGHT;

    game.state[(food_y * WIDTH) + food_x] = FOOD;
}

init_game ::() {
    init_console();

    for c:0..game.state.count-1 {game.state[c] = EMPTY;}

    

    head_x := (random_get() + WIDTH ) % WIDTH;
    head_y := (random_get() + HEIGHT ) % HEIGHT;

    game.head = (head_y * WIDTH) + head_x;
    game.tail = game.head - game.snake_length;



    game.state[game.head] = SNAKE;
    game.state[game.tail] = SNAKE;

    for i:0..game.snake_length -1 {
        game.state[game.head-i] = SNAKE ; // initial snake position
        game.flow[game.head-i -1] = game.head - i; // initial snake position
    }

    place_food();

    


    // SetConsoleActiveScreenBuffer(hBuffer[0]);
}


MINUS_ONE:: 4294967295; 

// dx_map: {Up, Right, Down, Left}
dx_map := u32.[0, 1, 0, MINUS_ONE]; 
dy_map := u32.[MINUS_ONE, 0, 1, 0];

update_game :: () {
    current_path_direction := game.direction; // The direction from the LAST frame

    // Now, apply the random direction change (which simulates a change of input)
    if(random_get_zero_to_one() > .9)
    {
        desired_direction  := cast(s8)(random_get()%4); // The new *requested* direction
        is_opposite_move := (abs(desired_direction - current_path_direction) == 2);

        if (!is_opposite_move) {
            game.direction = desired_direction; // Adopt the new, valid direction
        }
    }

    current_x := (game.head % WIDTH);
    current_y := (game.head / WIDTH);


    dx := dx_map[game.direction];
    dy := dy_map[game.direction];


    next_x := ((current_x + WIDTH + dx)) % WIDTH;
    next_y := ((current_y + HEIGHT + dy)) % HEIGHT;


    new_head := (next_y * WIDTH) + next_x ;

    
    if game.state[new_head] == SNAKE {
        init_game();
        Sleep(100);
    }

    bGrow := false;
    if new_head == game.food {
        game.score += 1;
        bGrow = true;
        place_food();
    }


    game.state[new_head] = SNAKE;



    // game.state[(tail_y * WIDTH) + tail_x] = EMPTY;


    if !bGrow {
        game.state[game.tail] = EMPTY;
        game.tail = game.flow[game.tail];
        game.flow[game.head] = new_head;
    }

    game.head = new_head;

    

}


main :: () {

    init_game();
    
    while true {
        update_game();

        draw_game();
        flip_buffers();
    }

    close_game();
}   



