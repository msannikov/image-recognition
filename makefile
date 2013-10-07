CC=g++
CFLAGS=-c -Wall
LDFLAGS=
OBJDIR=$(CURDIR)/obj
SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(addprefix $(OBJDIR)/,$(notdir $(SOURCES:.cpp=.o)))
EXECUTABLE=run

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

$(OBJDIR)/%.o: src/%.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR)/*.o $(EXECUTABLE)

$(OBJECTS) : | $(OBJDIR)

$(OBJDIR):
	test -d $(OBJDIR) || mkdir $(OBJDIR)

